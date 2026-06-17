from typing import Any, Dict, List, Optional, Tuple, Union

import lpips
import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from tqdm import tqdm

from .id_loss import ArcFaceIDLoss
from ..base.base_model import BaseModel
from ..embedders import ConditionerWrapper
from ..unets import DiffusersUNet2DCondWrapper, DiffusersUNet2DWrapper
from ..vae import AutoencoderKLDiffusers
from .lbm_config import LBMConfig
from ..face_condition_adapter import FaceConditionalAdapter
from ..ea_dists_loss import EADISTSLoss
from ..local_edge_loss import LocalEdgeLoss

class LBMModel(BaseModel):
    """This is the LBM class which defines the model.

    Args:

        config (LBMConfig):
            Configuration for the model

        denoiser (Union[DiffusersUNet2DWrapper, DiffusersTransformer2DWrapper]):
            Denoiser to use for the diffusion model. Defaults to None

        training_noise_scheduler (EulerDiscreteScheduler):
            Noise scheduler to use for training. Defaults to None

        sampling_noise_scheduler (EulerDiscreteScheduler):
            Noise scheduler to use for sampling. Defaults to None

        vae (AutoencoderKLDiffusers):
            VAE to use for the diffusion model. Defaults to None

        conditioner (ConditionerWrapper):
            Conditioner to use for the diffusion model. Defaults to None
    """

    @classmethod
    def load_from_config(cls, config: LBMConfig):
        return cls(config=config)

    def __init__(
        self,
        config: LBMConfig,
        denoiser: Union[
            DiffusersUNet2DWrapper,
            DiffusersUNet2DCondWrapper,
        ] = None,
        training_noise_scheduler: FlowMatchEulerDiscreteScheduler = None,
        sampling_noise_scheduler: FlowMatchEulerDiscreteScheduler = None,
        vae: AutoencoderKLDiffusers = None,
        conditioner: ConditionerWrapper = None,
    ):
        BaseModel.__init__(self, config)

        self.vae = vae
        self.denoiser = denoiser
        self.conditioner = conditioner
        self.sampling_noise_scheduler = sampling_noise_scheduler
        self.training_noise_scheduler = training_noise_scheduler
        self.timestep_sampling = config.timestep_sampling
        self.latent_loss_type = config.latent_loss_type
        self.latent_loss_weight = config.latent_loss_weight
        self.pixel_loss_type = config.pixel_loss_type
        self.pixel_loss_max_size = config.pixel_loss_max_size
        self.pixel_loss_weight = config.pixel_loss_weight
        # id loss
        self.id_loss_weight = getattr(config, "id_loss_weight", 0.0)
        self.id_loss_model_path = getattr(config, "id_loss_model_path", None)
        self.id_loss_crop = getattr(config, "id_loss_crop", True)

        self.logit_mean = config.logit_mean
        self.logit_std = config.logit_std
        self.prob = config.prob
        self.selected_timesteps = config.selected_timesteps
        self.source_key = config.source_key
        self.target_key = config.target_key
        self.mask_key = config.mask_key
        self.bridge_noise_sigma = config.bridge_noise_sigma

        self.num_iterations = nn.Parameter(
            torch.tensor(0, dtype=torch.float32), requires_grad=False
        )
        # face adapter
        self.use_face_adapter = getattr(config, "use_face_adapter", False)
        self.parse_key = getattr(config, "parse_key", "parse")
        self.parse_adapter_scale = getattr(config, "parse_adapter_scale", 1.0)
        self.parse_num_classes = getattr(config, "parse_num_classes", 19)
        self.parse_adapter_include_mid = getattr(config, "parse_adapter_include_mid", True)
        self.use_sketch_face_adapter = getattr(config, "use_sketch_face_adapter", False)
        self.sketch_key = getattr(config, "sketch_key", None)
        self.sketch_in_channels = getattr(config, "sketch_in_channels", 3)

        self.use_coarse_face_adapter = getattr(config, "use_coarse_face_adapter", False)
        self.coarse_face_key = getattr(config, "coarse_face_key", None)
        self.coarse_in_channels = getattr(config, "coarse_in_channels", 3)

        if self.use_face_adapter:
            self.face_adapter = FaceConditionalAdapter(
                parse_in_channels=self.parse_num_classes,
                sketch_in_channels=self.sketch_in_channels,
                coarse_in_channels=self.coarse_in_channels,
                use_parse=True,
                use_sketch=self.use_sketch_face_adapter,
                use_coarse=self.use_coarse_face_adapter,
                include_mid=self.parse_adapter_include_mid,
                zero_init=getattr(config, "parse_adapter_zero_init", True),
                condition_dropout=getattr(config, "parse_adapter_condition_dropout", 0.0),
                use_scale_gates=getattr(config, "parse_adapter_use_scale_gates", True),
                gate_init=getattr(config, "parse_adapter_gate_init", 1.0),
            )
        else:
            self.face_adapter = None

        self.lpips_loss = None
        self.ea_dists_loss = None

        self.use_bridge_maam = getattr(config, "use_bridge_maam", False)

        self.local_edge_loss_weight = getattr(config, "local_edge_loss_weight", 0.0)
        self.local_edge_parts = getattr(config, "local_edge_parts", None)
        self.local_edge_dilate_kernel = getattr(config, "local_edge_dilate_kernel", 7)
        self.local_edge_exclude_labels = getattr(config, "local_edge_exclude_labels", None)
        self.local_edge_exclude_dilate_kernel = getattr(
            config,
            "local_edge_exclude_dilate_kernel",
            7,
        )

        self.local_edge_loss = None
        if self.local_edge_loss_weight > 0:
            parts = self.local_edge_parts
            if parts is None:
                parts = ["eye"]

            self.local_edge_loss = LocalEdgeLoss(
                parts=parts,
                num_classes=self.parse_num_classes,
                dilate_kernel=self.local_edge_dilate_kernel,
                exclude_labels=self.local_edge_exclude_labels,
                exclude_dilate_kernel=self.local_edge_exclude_dilate_kernel,
            )


        if self.pixel_loss_weight > 0:
            if self.pixel_loss_type == "lpips":
                self.lpips_loss = lpips.LPIPS(net="vgg")

            elif self.pixel_loss_type in {"dists", "ea_dists"}:
                use_edge = (
                        self.pixel_loss_type == "ea_dists"
                        and getattr(config, "ea_dists_use_edge", True)
                )

                self.ea_dists_loss = EADISTSLoss(
                    edge_weight=getattr(config, "ea_dists_edge_weight", 1.0),
                    use_edge=use_edge,
                    edge_to_rgb=getattr(config, "ea_dists_edge_to_rgb", True),
                    edge_normalize=getattr(config, "ea_dists_edge_normalize", True),
                    resize_to=getattr(config, "ea_dists_resize_to", None),
                )

        if self.id_loss_weight > 0:
            self.id_loss = ArcFaceIDLoss(
                model_path=self.id_loss_model_path,
                crop_face=self.id_loss_crop,
            )
        else:
            self.id_loss = None

    def _prepare_image_condition_for_adapter(
            self,
            x,
            dtype,
            device,
            target_hw,
            expected_channels: int = 3,
            name: str = "image_condition",
    ):
        """
        Prepare image-like condition for FaceConditionalAdapter.

        Expected:
          x: [B,C,H,W]
          output: [B,expected_channels,target_h,target_w]
        """
        x = x.to(device=device, dtype=dtype)

        if x.ndim != 4:
            raise ValueError(
                f"{name} must be a 4D tensor [B,C,H,W], got shape={tuple(x.shape)}"
            )

        if x.shape[-2:] != target_hw:
            x = torch.nn.functional.interpolate(
                x,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )

        # Insurance: if a future dataset loads sketch as grayscale [B,1,H,W],
        # but the adapter stem expects RGB-like 3 channels.
        if x.shape[1] == 1 and expected_channels == 3:
            x = x.repeat(1, 3, 1, 1)

        if x.shape[1] != expected_channels:
            raise ValueError(
                f"{name} channel mismatch: expected {expected_channels}, "
                f"got {x.shape[1]}, shape={tuple(x.shape)}"
            )

        return x

    def _get_face_adapter_residuals(self, batch, dtype, device):
        if self.face_adapter is None:
            return None, None

        if self.parse_key not in batch:
            raise KeyError(
                f"use_face_adapter=True but parse_key='{self.parse_key}' not found in batch. "
                f"Available keys: {list(batch.keys())}"
            )

        target_hw = batch[self.target_key].shape[-2:]

        parse = self._prepare_parse_for_adapter(
            batch[self.parse_key],
            dtype=dtype,
            device=device,
        )

        sketch = None
        if self.use_sketch_face_adapter:
            sketch_key = self.sketch_key or self.source_key

            if sketch_key not in batch:
                raise KeyError(
                    f"use_sketch_face_adapter=True but sketch_key='{sketch_key}' not found in batch. "
                    f"Available keys: {list(batch.keys())}"
                )

            sketch = self._prepare_image_condition_for_adapter(
                batch[sketch_key],
                dtype=dtype,
                device=device,
                target_hw=target_hw,
                expected_channels=self.sketch_in_channels,
                name=f"sketch condition '{sketch_key}'",
            )

        coarse_face = None
        if self.use_coarse_face_adapter:
            if self.coarse_face_key is None or self.coarse_face_key not in batch:
                raise KeyError(
                    f"use_coarse_face_adapter=True but coarse_face_key='{self.coarse_face_key}' not found. "
                    f"Available keys: {list(batch.keys())}"
                )

            coarse_face = self._prepare_image_condition_for_adapter(
                batch[self.coarse_face_key],
                dtype=dtype,
                device=device,
                target_hw=target_hw,
                expected_channels=self.coarse_in_channels,
                name=f"coarse face condition '{self.coarse_face_key}'",
            )

        adapter_out = self.face_adapter(
            parse=parse,
            sketch=sketch,
            coarse_face=coarse_face,
        )

        scale = self.parse_adapter_scale
        down = [x * scale for x in adapter_out["down"]]

        mid = adapter_out.get("mid", None)
        if mid is not None:
            mid = mid * scale

        return down, mid



    def on_fit_start(self, device: torch.device | None = None, *args, **kwargs):
        """Called when the training starts"""
        super().on_fit_start(device=device, *args, **kwargs)
        if self.vae is not None:
            self.vae.on_fit_start(device=device, *args, **kwargs)
        if self.conditioner is not None:
            self.conditioner.on_fit_start(device=device, *args, **kwargs)

    def forward(self, batch: Dict[str, Any], step=0, batch_idx=0, *args, **kwargs):

        self.num_iterations += 1

        # Get inputs/latents
        if self.vae is not None:
            vae_inputs = batch[self.target_key]
            z = self.vae.encode(vae_inputs)
            downsampling_factor = self.vae.downsampling_factor
        else:
            z = batch[self.target_key]
            downsampling_factor = 1

        if self.mask_key in batch:
            valid_mask = batch[self.mask_key].bool()[:, 0, :, :].unsqueeze(1)
            invalid_mask = ~valid_mask
            valid_mask_for_latent = ~torch.max_pool2d(
                invalid_mask.float(),
                downsampling_factor,
                downsampling_factor,
            ).bool()
            valid_mask_for_latent = valid_mask_for_latent.repeat((1, z.shape[1], 1, 1))

        else:
            valid_mask = torch.ones_like(batch[self.target_key]).bool()
            valid_mask_for_latent = torch.ones_like(z).bool()

        source_image = batch[self.source_key]
        source_image = torch.nn.functional.interpolate(
            source_image,
            size=batch[self.target_key].shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).to(z.dtype)
        if self.vae is not None:
            z_source = self.vae.encode(source_image)

        else:
            z_source = source_image

        # Get conditionings
        conditioning = self._get_conditioning(batch, *args, **kwargs)

        # Sample a timestep
        timestep = self._timestep_sampling(n_samples=z.shape[0], device=z.device)
        sigmas = None

        # Create interpolant
        sigmas = self._get_sigmas(
            self.training_noise_scheduler, timestep, n_dim=4, device=z.device
        )
        noisy_sample = (
            sigmas * z_source
            + (1.0 - sigmas) * z
            + self.bridge_noise_sigma
            * (sigmas * (1.0 - sigmas)) ** 0.5
            * torch.randn_like(z)
        )

        for i, t in enumerate(timestep):
            if t.item() == self.training_noise_scheduler.timesteps[0]:
                noisy_sample[i] = z_source[i]

        # Predict noise level using denoiser
        adapter_down_residuals, adapter_mid_residual = self._get_face_adapter_residuals(
            batch=batch,
            dtype=noisy_sample.dtype,
            device=noisy_sample.device,
        )

        adapter_residual_norm = torch.zeros((), device=noisy_sample.device)
        adapter_down_norms = []

        if adapter_down_residuals is not None:
            adapter_norms = []

            for r in adapter_down_residuals:
                if r is not None:
                    n = r.detach().float().abs().mean()
                    adapter_norms.append(n)
                    adapter_down_norms.append(n)

            if adapter_mid_residual is not None:
                adapter_norms.append(adapter_mid_residual.detach().float().abs().mean())

            if len(adapter_norms) > 0:
                adapter_residual_norm = torch.stack(adapter_norms).mean()

        while len(adapter_down_norms) < 4:
            adapter_down_norms.append(torch.zeros((), device=noisy_sample.device))

        prediction = self.denoiser(
            sample=noisy_sample,
            timestep=timestep,
            conditioning=conditioning,
            down_intrablock_additional_residuals=adapter_down_residuals,
            mid_block_additional_residual=adapter_mid_residual,
            *args,
            **kwargs,
        )

        target = z_source - z
        denoised_sample = noisy_sample - prediction * sigmas
        target_pixels = batch[self.target_key]

        # Compute loss
        if self.latent_loss_weight > 0:
            loss = self.latent_loss(prediction, target.detach(), valid_mask_for_latent)
            latent_recon_loss = loss.mean()

        else:
            loss = torch.zeros(z.shape[0], device=z.device)
            latent_recon_loss = torch.zeros_like(loss)

        if (
                self.pixel_loss_weight > 0
                or self.id_loss_weight > 0
                or self.local_edge_loss_weight > 0
        ):
            denoised_sample = self._predicted_x_0(
                model_output=prediction,
                sample=noisy_sample,
                sigmas=sigmas,
            )

            parse_for_local_loss = None
            if self.local_edge_loss_weight > 0:
                if self.parse_key not in batch:
                    raise KeyError(
                        f"local_edge_loss_weight > 0 but parse_key='{self.parse_key}' not found. "
                        f"Available keys: {list(batch.keys())}"
                    )
                parse_for_local_loss = batch[self.parse_key]

            pixel_loss, id_recon_loss, local_edge_recon_loss = self.image_losses(
                denoised_sample,
                target_pixels.detach(),
                valid_mask,
                parse=parse_for_local_loss,
            )

            if self.pixel_loss_weight > 0:
                loss += self.pixel_loss_weight * pixel_loss

            if self.id_loss_weight > 0:
                loss += self.id_loss_weight * id_recon_loss

            if self.local_edge_loss_weight > 0:
                loss += self.local_edge_loss_weight * local_edge_recon_loss

        else:
            pixel_loss = torch.zeros_like(latent_recon_loss)
            id_recon_loss = torch.zeros_like(latent_recon_loss)
            local_edge_recon_loss = torch.zeros_like(latent_recon_loss)

        zero_log = torch.zeros((), device=loss.device)

        ea_dists_dists_loss = getattr(
            self,
            "_last_ea_dists_dists_loss",
            zero_log,
        )

        ea_dists_edge_loss = getattr(
            self,
            "_last_ea_dists_edge_loss",
            zero_log,
        )

        ea_dists_total_loss = getattr(
            self,
            "_last_ea_dists_total_loss",
            zero_log,
        )

        if hasattr(self.denoiser, "get_bridge_maam_log_dict"):
            bridge_maam_logs = self.denoiser.get_bridge_maam_log_dict(device=loss.device)
        else:
            bridge_maam_logs = {
                "bridge_maam_alpha_mean": torch.zeros((), device=loss.device),
                "bridge_maam_attn_mean": torch.zeros((), device=loss.device),
                "bridge_maam_attn_std": torch.zeros((), device=loss.device),
                "bridge_maam_delta_ratio": torch.zeros((), device=loss.device),
                "bridge_maam_direct_diff_ratio": torch.zeros((), device=loss.device),
            }

        return {
            "loss": loss.mean(),
            "latent_recon_loss": latent_recon_loss,
            "pixel_recon_loss": pixel_loss.mean(),
            "id_recon_loss": id_recon_loss.mean(),

            "local_edge_recon_loss": local_edge_recon_loss.mean(),
            "local_edge_mask_mean": getattr(
                self,
                "_last_local_edge_mask_mean",
                torch.zeros((), device=loss.device),
            ),
            "bridge_maam_alpha_mean": bridge_maam_logs["bridge_maam_alpha_mean"],
            "bridge_maam_attn_mean": bridge_maam_logs["bridge_maam_attn_mean"],
            "bridge_maam_attn_std": bridge_maam_logs["bridge_maam_attn_std"],
            "bridge_maam_delta_ratio": bridge_maam_logs["bridge_maam_delta_ratio"],
            "bridge_maam_direct_diff_ratio": bridge_maam_logs["bridge_maam_direct_diff_ratio"],

            "ea_dists_dists_loss": ea_dists_dists_loss,
            "ea_dists_edge_loss": ea_dists_edge_loss,
            "ea_dists_total_loss": ea_dists_total_loss,

            "adapter_residual_norm": adapter_residual_norm,
            "adapter_down0_norm": adapter_down_norms[0],
            "adapter_down1_norm": adapter_down_norms[1],
            "adapter_down2_norm": adapter_down_norms[2],
            "adapter_down3_norm": adapter_down_norms[3],
            "predicted_hr": denoised_sample,
            "noisy_sample": noisy_sample,
        }

    def latent_loss(self, prediction, model_input, valid_latent_mask):
        if self.latent_loss_type == "l2":
            return torch.mean(
                (
                    (prediction * valid_latent_mask - model_input * valid_latent_mask)
                    ** 2
                ).reshape(model_input.shape[0], -1),
                1,
            )
        elif self.latent_loss_type == "l1":
            return torch.mean(
                torch.abs(
                    prediction * valid_latent_mask - model_input * valid_latent_mask
                ).reshape(model_input.shape[0], -1),
                1,
            )
        else:
            raise NotImplementedError(
                f"Loss type {self.latent_loss_type} not implemented"
            )

    def image_losses(self, prediction, model_input, valid_mask, parse=None):
        latent_crop = self.pixel_loss_max_size // self.vae.downsampling_factor
        input_crop = self.pixel_loss_max_size

        crop_h = max((prediction.shape[2] - latent_crop), 0)
        crop_w = max((prediction.shape[3] - latent_crop), 0)

        input_crop_h = max((model_input.shape[2] - self.pixel_loss_max_size), 0)
        input_crop_w = max((model_input.shape[3] - self.pixel_loss_max_size), 0)

        # image random cropping
        if crop_h == 0:
            offset_h = 0
        else:
            offset_h = torch.randint(0, crop_h, (1,)).item()

        if crop_w == 0:
            offset_w = 0
        else:
            offset_w = torch.randint(0, crop_w, (1,)).item()

        input_offset_h = offset_h * self.vae.downsampling_factor
        input_offset_w = offset_w * self.vae.downsampling_factor

        prediction = prediction[
            :,
            :,
            crop_h - offset_h: min(
                crop_h - offset_h + latent_crop,
                prediction.shape[2],
            ),
            crop_w - offset_w: min(
                crop_w - offset_w + latent_crop,
                prediction.shape[3],
            ),
        ]

        model_input = model_input[
            :,
            :,
            input_crop_h - input_offset_h: min(
                input_crop_h - input_offset_h + input_crop,
                model_input.shape[2],
            ),
            input_crop_w - input_offset_w: min(
                input_crop_w - input_offset_w + input_crop,
                model_input.shape[3],
            ),
        ]

        valid_mask = valid_mask[
            :,
            :,
            input_crop_h - input_offset_h: min(
                input_crop_h - input_offset_h + input_crop,
                valid_mask.shape[2],
            ),
            input_crop_w - input_offset_w: min(
                input_crop_w - input_offset_w + input_crop,
                valid_mask.shape[3],
            ),
        ]

        # Keep parse spatially aligned with model_input / valid_mask.
        # In your current 256x256 setting this usually does not crop anything,
        # but this keeps the method correct if pixel_loss_max_size is smaller.
        if parse is not None:
            if parse.ndim == 4:
                parse = parse[
                    :,
                    :,
                    input_crop_h - input_offset_h: min(
                        input_crop_h - input_offset_h + input_crop,
                        parse.shape[2],
                    ),
                    input_crop_w - input_offset_w: min(
                        input_crop_w - input_offset_w + input_crop,
                        parse.shape[3],
                    ),
                ]
            elif parse.ndim == 3:
                parse = parse[
                    :,
                    input_crop_h - input_offset_h: min(
                        input_crop_h - input_offset_h + input_crop,
                        parse.shape[1],
                    ),
                    input_crop_w - input_offset_w: min(
                        input_crop_w - input_offset_w + input_crop,
                        parse.shape[2],
                    ),
                ]
            else:
                raise ValueError(
                    f"Unsupported parse shape in image_losses: {tuple(parse.shape)}"
                )

        decoded_prediction = self.vae.decode(prediction).clamp(-1, 1)

        # ---------- pixel loss ----------
        if self.pixel_loss_weight <= 0:
            pixel_loss = torch.zeros(model_input.shape[0], device=model_input.device)

        elif self.pixel_loss_type == "l2":
            pixel_loss = torch.mean(
                (
                        (decoded_prediction * valid_mask - model_input * valid_mask) ** 2
                ).reshape(model_input.shape[0], -1),
                1,
            )

        elif self.pixel_loss_type == "l1":
            pixel_loss = torch.mean(
                torch.abs(
                    decoded_prediction * valid_mask - model_input * valid_mask
                ).reshape(model_input.shape[0], -1),
                1,
            )

        elif self.pixel_loss_type == "lpips":
            pixel_loss = self.lpips_loss(
                decoded_prediction * valid_mask,
                model_input * valid_mask,
            ).mean()

        elif self.pixel_loss_type in {"dists", "ea_dists"}:
            if self.ea_dists_loss is None:
                raise RuntimeError(
                    f"pixel_loss_type={self.pixel_loss_type} "
                    f"but ea_dists_loss is not initialized."
                )

            pixel_loss, ea_logs = self.ea_dists_loss(
                decoded_prediction * valid_mask,
                model_input * valid_mask,
                return_dict=True,
            )

            self._last_ea_dists_dists_loss = ea_logs["dists"]
            self._last_ea_dists_edge_loss = ea_logs["edge"]
            self._last_ea_dists_total_loss = ea_logs["total"]

        else:
            raise NotImplementedError(
                f"Pixel loss type {self.pixel_loss_type} not implemented"
            )

        # ---------- local edge loss ----------
        if self.local_edge_loss_weight > 0 and self.local_edge_loss is not None:
            if parse is None:
                raise ValueError(
                    "parse is required when local_edge_loss_weight > 0"
                )

            local_edge_loss, local_edge_logs = self.local_edge_loss(
                pred=decoded_prediction,
                target=model_input,
                parse=parse,
                valid_mask=valid_mask,
                return_dict=True,
            )

            self._last_local_edge_loss = local_edge_logs["local_edge_loss"]
            self._last_local_edge_mask_mean = local_edge_logs[
                "local_edge_mask_mean"
            ]

        else:
            local_edge_loss = torch.zeros_like(pixel_loss.mean())
            self._last_local_edge_loss = torch.zeros_like(pixel_loss.mean())
            self._last_local_edge_mask_mean = torch.zeros_like(pixel_loss.mean())

        # ---------- ArcFace ID loss ----------
        if self.id_loss_weight > 0 and self.id_loss is not None:
            # ID loss should use face image itself, not masked image.
            # Input range remains [-1, 1], same as LPIPS.
            id_loss = self.id_loss(decoded_prediction, model_input)
        else:
            id_loss = torch.zeros_like(pixel_loss.mean())

        return pixel_loss, id_loss, local_edge_loss


    def _prepare_parse_for_adapter(self, parse, dtype, device):
        import torch.nn.functional as F

        raw_shape = tuple(parse.shape)
        raw_dtype = parse.dtype
        raw_min = parse.min().item()
        raw_max = parse.max().item()

        parse = parse.to(device=device)

        # Case 1: already one-hot
        if parse.ndim == 4 and parse.shape[1] == self.parse_num_classes:
            parse = parse.to(dtype=dtype)

            if parse.min() < -1e-4 or parse.max() > 1.0 + 1e-4:
                raise ValueError(
                    f"Parse looks like one-hot but value range is invalid: "
                    f"min={parse.min().item()}, max={parse.max().item()}"
                )

        else:
            if parse.ndim == 4 and parse.shape[1] == 1:
                parse = parse[:, 0]

            if parse.ndim != 3:
                raise ValueError(
                    f"Unsupported parse shape: {tuple(parse.shape)}. "
                    f"Expected [B,19,H,W], [B,1,H,W], or [B,H,W]."
                )

            pmin = parse.min().item()
            pmax = parse.max().item()

            if parse.dtype.is_floating_point and pmin < 0:
                parse = torch.round(((parse + 1.0) / 2.0) * 255.0)
            elif parse.dtype.is_floating_point and pmax <= 1.0:
                parse = torch.round(parse * 255.0)

            parse = parse.long()

            if parse.min() < 0 or parse.max() >= self.parse_num_classes:
                raise ValueError(
                    f"Invalid parse label range after conversion: "
                    f"shape={tuple(parse.shape)}, "
                    f"min={parse.min().item()}, max={parse.max().item()}, "
                    f"num_classes={self.parse_num_classes}"
                )

            parse = F.one_hot(
                parse,
                num_classes=self.parse_num_classes,
            ).permute(0, 3, 1, 2).contiguous()

            parse = parse.to(device=device, dtype=dtype)

        if not hasattr(self, "_debug_parse_once"):
            with torch.no_grad():
                label = parse.argmax(dim=1)
                print("\n[REAL TRAIN DEBUG parse before FaceAdapter]")
                print("raw shape:", raw_shape)
                print("raw dtype:", raw_dtype)
                print("raw min/max:", raw_min, raw_max)
                print("prepared shape:", tuple(parse.shape))
                print("prepared dtype:", parse.dtype)
                print("prepared min/max:", parse.min().item(), parse.max().item())
                print(
                    "channel_sum min/max:",
                    parse.sum(dim=1).min().item(),
                    parse.sum(dim=1).max().item(),
                )
                print(
                    "class_counts:",
                    torch.bincount(
                        label.flatten().detach().cpu(),
                        minlength=self.parse_num_classes,
                    ).tolist(),
                )
            self._debug_parse_once = True

        return parse

    # def _prepare_parse_for_adapter(self, parse, dtype, device):
    #     """
    #     Convert parse input to one-hot tensor for FaceConditionalAdapter.
    #
    #     Accept:
    #       [B, 19, H, W] one-hot
    #       [B, 1, H, W] label map
    #       [B, H, W] label map
    #
    #     Also recovers common wrong cases:
    #       ToTensor: label/255
    #       ToTensor + RescaleMapper: 2 * label/255 - 1
    #     """
    #     import torch.nn.functional as F
    #
    #     parse = parse.to(device=device)
    #
    #     # Case 1: already one-hot
    #     if parse.ndim == 4 and parse.shape[1] == self.parse_num_classes:
    #         parse = parse.to(dtype=dtype)
    #
    #         if parse.min() < -1e-4 or parse.max() > 1.0 + 1e-4:
    #             raise ValueError(
    #                 f"Parse looks like one-hot but value range is invalid: "
    #                 f"min={parse.min().item()}, max={parse.max().item()}"
    #             )
    #
    #         return parse
    #
    #     # Case 2: [B, 1, H, W] -> [B, H, W]
    #     if parse.ndim == 4 and parse.shape[1] == 1:
    #         parse = parse[:, 0]
    #
    #     if parse.ndim != 3:
    #         raise ValueError(
    #             f"Unsupported parse shape: {tuple(parse.shape)}. "
    #             f"Expected [B,19,H,W], [B,1,H,W], or [B,H,W]."
    #         )
    #
    #     pmin = parse.min().item()
    #     pmax = parse.max().item()
    #
    #     # Wrong case: ToTensor + RescaleMapper
    #     # label 0 -> -1
    #     # label 17 -> 2*(17/255)-1 = -0.8667
    #     if parse.dtype.is_floating_point and pmin < 0:
    #         parse = torch.round(((parse + 1.0) / 2.0) * 255.0)
    #
    #     # Wrong case: ToTensor only
    #     # label 17 -> 17/255 = 0.0667
    #     elif parse.dtype.is_floating_point and pmax <= 1.0:
    #         parse = torch.round(parse * 255.0)
    #
    #     parse = parse.long()
    #
    #     if parse.min() < 0 or parse.max() >= self.parse_num_classes:
    #         raise ValueError(
    #             f"Invalid parse label range after conversion: "
    #             f"shape={tuple(parse.shape)}, "
    #             f"min={parse.min().item()}, max={parse.max().item()}, "
    #             f"num_classes={self.parse_num_classes}"
    #         )
    #
    #     parse = F.one_hot(
    #         parse,
    #         num_classes=self.parse_num_classes,
    #     ).permute(0, 3, 1, 2).contiguous()
    #
    #     if not hasattr(self, "_debug_parse_once"):
    #         with torch.no_grad():
    #             label = parse.argmax(dim=1)
    #             print("\n[REAL TRAIN DEBUG parse before FaceAdapter]")
    #             print("shape:", tuple(parse.shape))
    #             print("dtype:", parse.dtype)
    #             print("min/max:", parse.min().item(), parse.max().item())
    #             print(
    #                 "channel_sum min/max:",
    #                 parse.sum(dim=1).min().item(),
    #                 parse.sum(dim=1).max().item(),
    #             )
    #             print(
    #                 "class_counts:",
    #                 torch.bincount(
    #                     label.flatten().detach().cpu(),
    #                     minlength=self.parse_num_classes,
    #                 ).tolist(),
    #             )
    #         self._debug_parse_once = True
    #
    #     return parse.to(device=device, dtype=dtype)



    def _get_conditioning(
        self,
        batch: Dict[str, Any],
        ucg_keys: List[str] = None,
        set_ucg_rate_zero=False,
        *args,
        **kwargs,
    ):
        """
        Get the conditionings
        """
        if self.conditioner is not None:
            return self.conditioner(
                batch,
                ucg_keys=ucg_keys,
                set_ucg_rate_zero=set_ucg_rate_zero,
                vae=self.vae,
                *args,
                **kwargs,
            )
        else:
            return None

    def _timestep_sampling(self, n_samples=1, device="cpu"):
        if self.timestep_sampling == "uniform":
            idx = torch.randint(
                0,
                self.training_noise_scheduler.config.num_train_timesteps,
                (n_samples,),
                device="cpu",
            )
            return self.training_noise_scheduler.timesteps[idx].to(device=device)

        elif self.timestep_sampling == "log_normal":
            u = torch.normal(
                mean=self.logit_mean,
                std=self.logit_std,
                size=(n_samples,),
                device="cpu",
            )
            u = torch.nn.functional.sigmoid(u)
            indices = (
                u * self.training_noise_scheduler.config.num_train_timesteps
            ).long()
            return self.training_noise_scheduler.timesteps[indices].to(device=device)

        elif self.timestep_sampling == "custom_timesteps":
            idx = np.random.choice(len(self.selected_timesteps), n_samples, p=self.prob)

            return torch.tensor(
                self.selected_timesteps, device=device, dtype=torch.long
            )[idx]

    def _predicted_x_0(
        self,
        model_output,
        sample,
        sigmas=None,
    ):
        """
        Predict x_0, the orinal denoised sample, using the model output and the timesteps depending on the prediction type.
        """
        pred_x_0 = sample - model_output * sigmas
        return pred_x_0

    def _get_sigmas(
        self, scheduler, timesteps, n_dim=4, dtype=torch.float32, device="cpu"
    ):
        sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    @torch.no_grad()
    def sample(
        self,
        z: torch.Tensor,
        num_steps: int = 20,
        conditioner_inputs: Optional[Dict[str, Any]] = None,
        max_samples: Optional[int] = None,
        verbose: bool = False,
    ):
        self.sampling_noise_scheduler.set_timesteps(
            sigmas=np.linspace(1, 1 / num_steps, num_steps)
        )

        sample = z

        # Get conditioning
        conditioning = self._get_conditioning(
            conditioner_inputs, set_ucg_rate_zero=True, device=z.device
        )

        adapter_down_residuals, adapter_mid_residual = self._get_face_adapter_residuals(
            batch=conditioner_inputs,
            dtype=z.dtype,
            device=z.device,
        )

        # If max_samples parameter is provided, limit the number of samples
        if max_samples is not None:
            sample = sample[:max_samples]

        if conditioning:
            conditioning["cond"] = {
                k: v[:max_samples] for k, v in conditioning["cond"].items()
            }

        for i, t in tqdm(
            enumerate(self.sampling_noise_scheduler.timesteps), disable=not verbose
        ):
            if hasattr(self.sampling_noise_scheduler, "scale_model_input"):
                denoiser_input = self.sampling_noise_scheduler.scale_model_input(
                    sample, t
                )

            else:
                denoiser_input = sample

            # Predict noise level using denoiser using conditionings
            pred = self.denoiser(
                sample=denoiser_input,
                timestep=t.to(z.device).repeat(denoiser_input.shape[0]),
                conditioning=conditioning,
                down_intrablock_additional_residuals=adapter_down_residuals,
                mid_block_additional_residual=adapter_mid_residual,
            )

            # Make one step on the reverse diffusion process
            sample = self.sampling_noise_scheduler.step(
                pred, t, sample, return_dict=False
            )[0]
            if i < len(self.sampling_noise_scheduler.timesteps) - 1:
                timestep = (
                    self.sampling_noise_scheduler.timesteps[i + 1]
                    .to(z.device)
                    .repeat(sample.shape[0])
                )
                sigmas = self._get_sigmas(
                    self.sampling_noise_scheduler, timestep, n_dim=4, device=z.device
                )
                sample = sample + self.bridge_noise_sigma * (
                    sigmas * (1.0 - sigmas)
                ) ** 0.5 * torch.randn_like(sample)
                sample = sample.to(z.dtype)

        if self.vae is not None:
            decoded_sample = self.vae.decode(sample)

        else:
            decoded_sample = sample

        return decoded_sample

    def log_samples(
        self,
        batch: Dict[str, Any],
        input_shape: Optional[Tuple[int, int, int]] = None,
        max_samples: Optional[int] = None,
        num_steps: Union[int, List[int]] = 20,
    ):
        if isinstance(num_steps, int):
            num_steps = [num_steps]

        logs = {}

        N = max_samples if max_samples is not None else len(batch[self.source_key])

        batch = {k: v[:N] for k, v in batch.items()}

        # infer input shape based on VAE configuration if not passed
        if input_shape is None:
            if self.vae is not None:
                # get input pixel size of the vae
                input_shape = batch[self.target_key].shape[2:]
                # rescale to latent size
                input_shape = (
                    self.vae.latent_channels,
                    input_shape[0] // self.vae.downsampling_factor,
                    input_shape[1] // self.vae.downsampling_factor,
                )
            else:
                raise ValueError(
                    "input_shape must be passed when no VAE is used in the model"
                )

        for num_step in num_steps:
            source_image = batch[self.source_key]
            source_image = torch.nn.functional.interpolate(
                source_image,
                size=batch[self.target_key].shape[2:],
                mode="bilinear",
                align_corners=False,
            ).to(dtype=self.dtype)
            if self.vae is not None:
                z = self.vae.encode(source_image)

            else:
                z = source_image

            with torch.autocast(dtype=self.dtype, device_type="cuda"):
                logs[f"samples_{num_step}_steps"] = self.sample(
                    z,
                    num_steps=num_step,
                    conditioner_inputs=batch,
                    max_samples=N,
                )

        return logs
