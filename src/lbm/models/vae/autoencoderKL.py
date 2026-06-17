import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models import AutoencoderKL

from ..base.base_model import BaseModel
from ..utils import Tiler, pad
from .autoencoderKL_config import AutoencoderKLDiffusersConfig


def _vae_encoder_forward_with_skips(self, sample: torch.Tensor) -> torch.Tensor:
    """
    VAE encoder forward with intermediate feature caching.

    This follows the first-stage skip connection idea in img2img-turbo:
    save the activations before each VAE encoder down block, then let the VAE
    decoder use them through 1x1 zero-conv skip connections.
    """
    sample = self.conv_in(sample)

    down_features = []
    for down_block in self.down_blocks:
        # Store the feature before each down block.
        # The VAE encoder is frozen, so detaching reduces graph/memory usage.
        down_features.append(sample.detach())
        sample = down_block(sample)

    sample = self.mid_block(sample)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)

    self.current_down_blocks = down_features
    return sample


def _vae_decoder_forward_with_skips(self, sample: torch.Tensor, latent_embeds=None) -> torch.Tensor:
    """
    VAE decoder forward with first-stage skip connections.

    For each decoder up block:
        sample = sample + gamma * skip_conv_i(encoder_feature_i)
        sample = up_block(sample)
    """
    sample = self.conv_in(sample)

    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype

    # middle
    sample = self.mid_block(sample, latent_embeds)
    sample = sample.to(upscale_dtype)

    use_skip = (
        not getattr(self, "ignore_skip", True)
        and hasattr(self, "incoming_skip_acts")
        and self.incoming_skip_acts is not None
        and hasattr(self, "skip_conv_1")
        and hasattr(self, "skip_conv_2")
        and hasattr(self, "skip_conv_3")
        and hasattr(self, "skip_conv_4")
    )

    if use_skip:
        skip_convs = [
            self.skip_conv_1,
            self.skip_conv_2,
            self.skip_conv_3,
            self.skip_conv_4,
        ]

        # Encoder features are shallow -> deep, decoder uses deep -> shallow.
        skip_acts = self.incoming_skip_acts[::-1]
        gamma = float(getattr(self, "gamma", 1.0))

        for idx, up_block in enumerate(self.up_blocks):
            if idx < len(skip_acts) and idx < len(skip_convs):
                skip = skip_acts[idx]

                if skip.shape[-2:] != sample.shape[-2:]:
                    skip = F.interpolate(
                        skip,
                        size=sample.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                skip = skip.to(device=sample.device, dtype=sample.dtype)
                sample = sample + gamma * skip_convs[idx](skip)

            sample = up_block(sample, latent_embeds)

    else:
        for up_block in self.up_blocks:
            sample = up_block(sample, latent_embeds)

    # post-process
    if latent_embeds is None:
        sample = self.conv_norm_out(sample)
    else:
        sample = self.conv_norm_out(sample, latent_embeds)

    sample = self.conv_act(sample)
    sample = self.conv_out(sample)

    return sample


class AutoencoderKLDiffusers(BaseModel):
    """This is the VAE class used to work with latent models

    Args:

        config (AutoencoderKLDiffusersConfig): The config class which defines all the required parameters.
    """

    def __init__(self, config: AutoencoderKLDiffusersConfig):
        BaseModel.__init__(self, config)
        self.config = config
        self.vae_model = AutoencoderKL.from_pretrained(
            config.version,
            subfolder=config.subfolder,
            revision=config.revision,
        )
        self.tiling_size = config.tiling_size
        self.tiling_overlap = config.tiling_overlap

        self.use_vae_skip = getattr(config, "use_vae_skip", False)
        self.vae_skip_zero_init = getattr(config, "vae_skip_zero_init", True)
        self.vae_skip_gamma = getattr(config, "vae_skip_gamma", 1.0)

        # get downsampling factor
        self._get_properties()

        if self.use_vae_skip:
            self.enable_vae_skip(
                zero_init=self.vae_skip_zero_init,
                gamma=self.vae_skip_gamma,
            )

    @torch.no_grad()
    def _get_properties(self):
        self.has_shift_factor = (
            hasattr(self.vae_model.config, "shift_factor")
            and self.vae_model.config.shift_factor is not None
        )
        self.shift_factor = (
            self.vae_model.config.shift_factor if self.has_shift_factor else 0
        )

        # set latent channels
        self.latent_channels = self.vae_model.config.latent_channels
        self.has_latents_mean = (
            hasattr(self.vae_model.config, "latents_mean")
            and self.vae_model.config.latents_mean is not None
        )
        self.has_latents_std = (
            hasattr(self.vae_model.config, "latents_std")
            and self.vae_model.config.latents_std is not None
        )
        self.latents_mean = self.vae_model.config.latents_mean
        self.latents_std = self.vae_model.config.latents_std

        x = torch.randn(1, self.vae_model.config.in_channels, 32, 32)
        z = self.encode(x)

        # set downsampling factor
        self.downsampling_factor = int(x.shape[2] / z.shape[2])

    def enable_vae_skip(self, zero_init: bool = True, gamma: float = 1.0):
        """
        Enable VAE encoder-decoder skip connections.

        This follows the img2img-turbo first-stage skip idea:
            encoder intermediate features -> 1x1 zero-conv -> decoder up blocks

        For SD-style VAE block_out_channels [128, 256, 512, 512], this creates:
            skip_conv_1: 512 -> 512
            skip_conv_2: 256 -> 512
            skip_conv_3: 128 -> 512
            skip_conv_4: 128 -> 256
        """
        self.use_vae_skip = True

        self.vae_model.encoder.forward = _vae_encoder_forward_with_skips.__get__(
            self.vae_model.encoder,
            self.vae_model.encoder.__class__,
        )
        self.vae_model.decoder.forward = _vae_decoder_forward_with_skips.__get__(
            self.vae_model.decoder,
            self.vae_model.decoder.__class__,
        )

        block_out = list(self.vae_model.config.block_out_channels)
        if len(block_out) != 4:
            raise ValueError(
                "[VAE-Skip] This implementation expects 4 VAE block_out_channels, "
                f"but got {block_out}."
            )

        # Encoder cached features before each down block:
        # [block_out[0], block_out[0], block_out[1], block_out[2]]
        # Reversed for decoder:
        # [block_out[2], block_out[1], block_out[0], block_out[0]]
        skip_in_channels = [
            block_out[2],
            block_out[1],
            block_out[0],
            block_out[0],
        ]

        # Decoder feature channels before each up block:
        # [block_out[3], block_out[2], block_out[2], block_out[1]]
        skip_out_channels = [
            block_out[3],
            block_out[2],
            block_out[2],
            block_out[1],
        ]

        skip_convs = []
        for in_c, out_c in zip(skip_in_channels, skip_out_channels):
            conv = nn.Conv2d(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            if zero_init:
                nn.init.zeros_(conv.weight)
            else:
                nn.init.constant_(conv.weight, 1e-5)
            skip_convs.append(conv)

        self.vae_model.decoder.skip_conv_1 = skip_convs[0]
        self.vae_model.decoder.skip_conv_2 = skip_convs[1]
        self.vae_model.decoder.skip_conv_3 = skip_convs[2]
        self.vae_model.decoder.skip_conv_4 = skip_convs[3]

        self.vae_model.decoder.ignore_skip = False
        self.vae_model.decoder.gamma = float(gamma)
        self.vae_model.decoder.incoming_skip_acts = None

        print(
            "[VAE-Skip] enabled:",
            f"zero_init={zero_init}, gamma={gamma},",
            f"skip_in={skip_in_channels}, skip_out={skip_out_channels}",
        )

    def enable_vae_skip_trainable(self):
        """
        Freeze the pretrained VAE and only train the newly added skip convs.

        Call this after vae.freeze() in the training script.
        """
        if not getattr(self, "use_vae_skip", False):
            return

        for p in self.vae_model.parameters():
            p.requires_grad = False

        decoder = self.vae_model.decoder
        for name in ["skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4"]:
            if hasattr(decoder, name):
                for p in getattr(decoder, name).parameters():
                    p.requires_grad = True

    def _prepare_decoder_skips(self, z: torch.Tensor):
        """
        Pass cached encoder features to the decoder before decoding.

        In the current LBM forward order, source image is encoded after target image,
        so the cached encoder features are from the source sketch.
        """
        if not getattr(self, "use_vae_skip", False):
            return

        encoder = self.vae_model.encoder
        decoder = self.vae_model.decoder

        if not hasattr(encoder, "current_down_blocks"):
            decoder.incoming_skip_acts = None
            decoder.ignore_skip = True
            return

        skip_acts = encoder.current_down_blocks

        if len(skip_acts) == 0 or skip_acts[0].shape[0] != z.shape[0]:
            decoder.incoming_skip_acts = None
            decoder.ignore_skip = True
            return

        decoder.incoming_skip_acts = skip_acts
        decoder.ignore_skip = False

    def encode(self, x: torch.tensor, batch_size: int = 8):
        latents = []
        cached_down_blocks = None

        for i in range(0, x.shape[0], batch_size):
            latents.append(
                self.vae_model.encode(x[i : i + batch_size]).latent_dist.sample()
            )

            if getattr(self, "use_vae_skip", False) and hasattr(
                self.vae_model.encoder,
                "current_down_blocks",
            ):
                current = self.vae_model.encoder.current_down_blocks
                if cached_down_blocks is None:
                    cached_down_blocks = [[] for _ in range(len(current))]

                for level_idx, feat in enumerate(current):
                    cached_down_blocks[level_idx].append(feat)

        if cached_down_blocks is not None:
            self.vae_model.encoder.current_down_blocks = [
                torch.cat(level_feats, dim=0)
                for level_feats in cached_down_blocks
            ]

        latents = torch.cat(latents, dim=0)
        latents = (latents - self.shift_factor) * self.vae_model.config.scaling_factor

        return latents

    def decode(self, z: torch.tensor):

        if self.has_latents_mean and self.has_latents_std:
            latents_mean = (
                torch.tensor(self.latents_mean)
                .view(1, self.latent_channels, 1, 1)
                .to(z.device, z.dtype)
            )
            latents_std = (
                torch.tensor(self.latents_std)
                .view(1, self.latent_channels, 1, 1)
                .to(z.device, z.dtype)
            )
            z = z * latents_std / self.vae_model.config.scaling_factor + latents_mean
        else:
            z = z / self.vae_model.config.scaling_factor + self.shift_factor

        use_tiling = (
            z.shape[2] > self.tiling_size[0] or z.shape[3] > self.tiling_size[1]
        )

        if use_tiling:
            if getattr(self, "use_vae_skip", False):
                self.vae_model.decoder.incoming_skip_acts = None
                self.vae_model.decoder.ignore_skip = True

            samples = []
            for i in range(z.shape[0]):

                z_i = z[i].unsqueeze(0)

                tiler = Tiler()
                tiles = tiler.get_tiles(
                    input=z_i,
                    tile_size=self.tiling_size,
                    overlap_size=self.tiling_overlap,
                    scale=self.downsampling_factor,
                    out_channels=3,
                )

                for i, tile_row in enumerate(tiles):
                    for j, tile in enumerate(tile_row):
                        tile_shape = tile.shape
                        # pad tile to inference size if tile is smaller than inference size
                        tile = pad(
                            tile,
                            base_h=self.tiling_size[0],
                            base_w=self.tiling_size[1],
                        )
                        tile_decoded = self.vae_model.decode(tile).sample
                        tiles[i][j] = (
                            tile_decoded[
                                0,
                                :,
                                : int(tile_shape[2] * self.downsampling_factor),
                                : int(tile_shape[3] * self.downsampling_factor),
                            ]
                            .cpu()
                            .unsqueeze(0)
                        )

                # merge tiles
                samples.append(tiler.merge_tiles(tiles=tiles))

            samples = torch.cat(samples, dim=0)

        else:
            self._prepare_decoder_skips(z)
            samples = self.vae_model.decode(z).sample

        return samples
