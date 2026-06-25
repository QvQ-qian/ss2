import datetime
import logging
import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import datetime
import logging
import os
import random
import re
import shutil
import json
import subprocess
import time
import sys
from typing import List, Optional


import swanlab
import numpy as np
import torch.nn.functional as F

from PIL import Image
from torchvision.utils import make_grid
from torchvision.transforms import functional as TF
from pytorch_lightning.callbacks import Callback
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.image.fid import FrechetInceptionDistance
import lpips

import braceexpand
import fire
import torch
import yaml
# from diffusers import FlowMatchEulerDiscreteScheduler, StableDiffusionXLPipeline
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models import UNet2DConditionModel
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.resnet import ResnetBlock2D
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, Callback
from pytorch_lightning.strategies import FSDPStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torchvision.transforms import InterpolationMode

from lbm.data.datasets import DataModule, DataModuleConfig
from lbm.data.filters import KeyFilter, KeyFilterConfig
from lbm.data.mappers import (
    KeyRenameMapper,
    KeyRenameMapperConfig,
    MapperWrapper,
    RescaleMapper,
    RescaleMapperConfig,
    TorchvisionMapper,
    TorchvisionMapperConfig,
)
from lbm.models.embedders import (
    ConditionerWrapper,
    LatentsConcatEmbedder,
    LatentsConcatEmbedderConfig,
)
from lbm.models.lbm import LBMConfig, LBMModel
from lbm.models.unets import DiffusersUNet2DCondWrapper
from lbm.models.vae import AutoencoderKLDiffusers, AutoencoderKLDiffusersConfig
from lbm.trainer import TrainingConfig, TrainingPipeline
# from lbm.trainer.loggers import WandbSampleLogger
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.strategies import FSDPStrategy

from swanlab.integration.pytorch_lightning import SwanLabLogger
from lbm.trainer.utils import StateDictAdapter


def get_model(
    backbone_signature: str = "/root/dataset/weights/sd15-lite",
    vae_num_channels: int = 4,
    unet_input_channels: int = 4,
    timestep_sampling: str = "log_normal",
    selected_timesteps: Optional[List[float]] = None,
    prob: Optional[List[float]] = None,
    conditioning_images_keys: Optional[List[str]] = [],
    conditioning_masks_keys: Optional[List[str]] = [],
    source_key: str = "source_image",
    target_key: str = "source_image_paste",
    mask_key: str = "mask",
    bridge_noise_sigma: float = 0.0,
    # bidirectional LBM
    bidirectional: bool = False,
    bidirectional_mode: str = "none",
    direction_aware: bool = False,
    num_directions: int = 2,
    direction_embed_init: float = 0.0,
    reverse_loss_weight: float = 0.5,
    reverse_use_pixel_loss: bool = False,
    reverse_pixel_loss_type: str = None,
    reverse_pixel_loss_weight: float = None,
    reverse_use_id_loss: bool = False,
    reverse_use_local_edge_loss: bool = False,
    eval_directions: Optional[List[str]] = None,
    eval_save_p2s_images: bool = False,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    pixel_loss_type: str = "lpips",
    latent_loss_type: str = "l2",
    latent_loss_weight: float = 1.0,
    pixel_loss_weight: float = 0.0,
    ea_dists_edge_weight: float = 1.0,
    ea_dists_use_edge: bool = True,
    ea_dists_edge_to_rgb: bool = True,
    ea_dists_edge_normalize: bool = True,
    ea_dists_resize_to: int = None,
    local_edge_loss_weight: float = 0.0,
    local_edge_parts: Optional[List[str]] = None,
    local_edge_dilate_kernel: int = 7,
    local_edge_exclude_labels: Optional[List[int]] = None,
    local_edge_exclude_dilate_kernel: int = 7,
    id_loss_weight: float = 0.0,
    id_loss_model_path: str = None,
    id_loss_crop: bool = True,
    # bridge-aware MAAM skip refinement
    use_bridge_maam: bool = False,
    bridge_maam_mode: str = "residual",
    bridge_maam_levels: Optional[List[str]] = None,
    bridge_maam_attn_type: str = "scsa",
    bridge_maam_alpha_init: float = 0.01,
    bridge_maam_attn_bias_init: float = 2.0,
    bridge_maam_use_timestep: bool = True,
    bridge_maam_zero_init_timestep: bool = True,
    bridge_maam_scsa_groups: int = 4,
    bridge_maam_scsa_kernels: Optional[List[int]] = None,
    bridge_maam_scsa_pool_size: int = 7,
    # vae skip
    use_vae_skip: bool = False,
    vae_skip_zero_init: bool = True,
    vae_skip_gamma: float = 1.0,
    # face adapter
    use_face_adapter: bool = False,
    parse_key: str = "parse",
    parse_num_classes: int = 19,
    parse_adapter_scale: float = 1.0,
    parse_adapter_condition_dropout: float = 0.0,
    parse_adapter_include_mid: bool = True,
    parse_adapter_zero_init: bool = True,
    parse_adapter_use_scale_gates: bool = True,
    parse_adapter_gate_init: float = 1.0,
    use_sketch_face_adapter: bool = False,
    sketch_key: str = None,
    sketch_in_channels: int = 3,
    use_coarse_face_adapter: bool = False,
    coarse_face_key: str = None,
    coarse_in_channels: int = 3,
):

    conditioners = []

    # Load pretrained model as base
    # pipe = StableDiffusionXLPipeline.from_pretrained(
    #     backbone_signature,
    #     torch_dtype=torch.bfloat16,
    # )

    ### MMMDiT ###
    # Get Architecture
    ### SD 1.5 UNet architecture ###
    denoiser = DiffusersUNet2DCondWrapper(
        in_channels=unet_input_channels,
        out_channels=vae_num_channels,
        center_input_sample=False,
        flip_sin_to_cos=True,
        freq_shift=0,
        down_block_types=[
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ],
        mid_block_type="UNetMidBlock2DCrossAttn",
        up_block_types=[
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ],
        only_cross_attention=False,
        block_out_channels=[320, 640, 1280, 1280],
        layers_per_block=2,
        downsample_padding=1,
        mid_block_scale_factor=1,
        dropout=0.0,
        act_fn="silu",
        norm_num_groups=32,
        norm_eps=1e-05,
        # cross_attention_dim=768,
        cross_attention_dim=[320, 640, 1280, 1280],
        transformer_layers_per_block=1,
        reverse_transformer_layers_per_block=None,
        encoder_hid_dim=None,
        encoder_hid_dim_type=None,
        attention_head_dim=8,
        num_attention_heads=None,
        dual_cross_attention=False,
        use_linear_projection=False,
        class_embed_type=None,
        addition_embed_type=None,
        addition_time_embed_dim=None,
        num_class_embeds=None,
        upcast_attention=False,
        resnet_time_scale_shift="default",
        resnet_skip_time_act=False,
        resnet_out_scale_factor=1.0,
        time_embedding_type="positional",
        time_embedding_dim=None,
        time_embedding_act_fn=None,
        timestep_post_act=None,
        time_cond_proj_dim=None,
        conv_in_kernel=3,
        conv_out_kernel=3,
        projection_class_embeddings_input_dim=None,
        attention_type="default",
        class_embeddings_concat=False,
        mid_block_only_cross_attention=None,
        cross_attention_norm=None,
        addition_embed_type_num_heads=64,
    ).to(torch.bfloat16)

    # state_dict = pipe.unet.state_dict()
    sd_unet = UNet2DConditionModel.from_pretrained(
        backbone_signature,
        subfolder="unet",
        torch_dtype=torch.bfloat16,
        # variant="fp16",
        use_safetensors=True,
        local_files_only=True,
    )

    state_dict = sd_unet.state_dict()

    for k in [
        "add_embedding.linear_1.weight",
        "add_embedding.linear_1.bias",
        "add_embedding.linear_2.weight",
        "add_embedding.linear_2.bias",
    ]:
        if k in state_dict:
            del state_dict[k]

    # Adapt the shapes
    state_dict_adapter = StateDictAdapter()
    state_dict = state_dict_adapter(
        model_state_dict=denoiser.state_dict(),
        checkpoint_state_dict=state_dict,
        regex_keys=[
            r"class_embedding.linear_\d+.(weight|bias)",
            r"conv_in.weight",
            r"(down_blocks|up_blocks)\.\d+\.attentions\.\d+\.transformer_blocks\.\d+\.attn\d+\.(to_k|to_v)\.weight",
            r"mid_block\.attentions\.\d+\.transformer_blocks\.\d+\.attn\d+\.(to_k|to_v)\.weight",
        ],
        strategy="zeros",
    )

    denoiser.load_state_dict(state_dict, strict=True)

    # Enable direction embedding after strict SD1.5 UNet weight loading.
    # This keeps SD1.5 pretrained loading unchanged.
    if direction_aware:
        denoiser.enable_direction_embedding(
            num_directions=num_directions,
            init_scale=direction_embed_init,
        )

    # Enable Bridge-aware MAAM after strict SD1.5 UNet weight loading.
    # This avoids breaking strict=True loading because MAAM parameters are newly added.
    if use_bridge_maam:
        if bridge_maam_levels is None:
            bridge_maam_levels = ["high"]

        if bridge_maam_scsa_kernels is None:
            bridge_maam_scsa_kernels = [3, 5, 7, 9]

        denoiser.enable_bridge_maam(
            mode=bridge_maam_mode,
            levels=bridge_maam_levels,
            attn_type=bridge_maam_attn_type,
            alpha_init=bridge_maam_alpha_init,
            attn_bias_init=bridge_maam_attn_bias_init,
            use_timestep=bridge_maam_use_timestep,
            zero_init_timestep=bridge_maam_zero_init_timestep,
            scsa_groups=bridge_maam_scsa_groups,
            scsa_kernels=bridge_maam_scsa_kernels,
            scsa_pool_size=bridge_maam_scsa_pool_size,
        )

        # New modules should follow the denoiser dtype.
        denoiser.to(torch.bfloat16)

    if direction_aware or use_bridge_maam:
        denoiser.to(torch.bfloat16)

    # del pipe
    del sd_unet

    if conditioning_images_keys != [] or conditioning_masks_keys != []:

        latents_concat_embedder_config = LatentsConcatEmbedderConfig(
            image_keys=conditioning_images_keys,
            mask_keys=conditioning_masks_keys,
        )
        latent_concat_embedder = LatentsConcatEmbedder(latents_concat_embedder_config)
        latent_concat_embedder.freeze()
        conditioners.append(latent_concat_embedder)

    # Wrap conditioners and set to device
    conditioner = ConditionerWrapper(
        conditioners=conditioners,
    )

    ## VAE ##
    # Get VAE model
    vae_config = AutoencoderKLDiffusersConfig(
        version=backbone_signature,
        subfolder="vae",
        tiling_size=(128, 128),
        use_vae_skip=use_vae_skip,
        vae_skip_zero_init=vae_skip_zero_init,
        vae_skip_gamma=vae_skip_gamma,
    )
    vae = AutoencoderKLDiffusers(vae_config)
    vae.freeze()

    if use_vae_skip:
        vae.enable_vae_skip_trainable()

    vae.to(torch.bfloat16)

    # LBM Config
    config = LBMConfig(
        ucg_keys=None,
        source_key=source_key,
        target_key=target_key,
        mask_key=mask_key,
        latent_loss_weight=latent_loss_weight,
        latent_loss_type=latent_loss_type,
        pixel_loss_type=pixel_loss_type,
        pixel_loss_weight=pixel_loss_weight,
        ea_dists_edge_weight=ea_dists_edge_weight,
        ea_dists_use_edge=ea_dists_use_edge,
        ea_dists_edge_to_rgb=ea_dists_edge_to_rgb,
        ea_dists_edge_normalize=ea_dists_edge_normalize,
        ea_dists_resize_to=ea_dists_resize_to,
        local_edge_loss_weight=local_edge_loss_weight,
        local_edge_parts=local_edge_parts,
        local_edge_dilate_kernel=local_edge_dilate_kernel,
        local_edge_exclude_labels=local_edge_exclude_labels,
        local_edge_exclude_dilate_kernel=local_edge_exclude_dilate_kernel,
        bidirectional=bidirectional,
        bidirectional_mode=bidirectional_mode,
        direction_aware=direction_aware,
        num_directions=num_directions,
        direction_embed_init=direction_embed_init,
        reverse_loss_weight=reverse_loss_weight,
        reverse_use_pixel_loss=reverse_use_pixel_loss,
        reverse_pixel_loss_type=reverse_pixel_loss_type,
        reverse_pixel_loss_weight=reverse_pixel_loss_weight,
        reverse_use_id_loss=reverse_use_id_loss,
        reverse_use_local_edge_loss=reverse_use_local_edge_loss,
        eval_directions=eval_directions,
        eval_save_p2s_images=eval_save_p2s_images,
        use_bridge_maam=use_bridge_maam,
        bridge_maam_mode=bridge_maam_mode,
        bridge_maam_levels=bridge_maam_levels,
        bridge_maam_attn_type=bridge_maam_attn_type,
        bridge_maam_alpha_init=bridge_maam_alpha_init,
        bridge_maam_attn_bias_init=bridge_maam_attn_bias_init,
        bridge_maam_use_timestep=bridge_maam_use_timestep,
        bridge_maam_zero_init_timestep=bridge_maam_zero_init_timestep,
        bridge_maam_scsa_groups=bridge_maam_scsa_groups,
        bridge_maam_scsa_kernels=bridge_maam_scsa_kernels,
        bridge_maam_scsa_pool_size=bridge_maam_scsa_pool_size,
        id_loss_weight=id_loss_weight,
        id_loss_model_path=id_loss_model_path,
        id_loss_crop=id_loss_crop,
        timestep_sampling=timestep_sampling,
        logit_mean=logit_mean,
        logit_std=logit_std,
        selected_timesteps=selected_timesteps,
        prob=prob,
        bridge_noise_sigma=bridge_noise_sigma,
        #face adapter
        use_face_adapter=use_face_adapter,
        parse_key=parse_key,
        parse_num_classes=parse_num_classes,
        parse_adapter_scale=parse_adapter_scale,
        parse_adapter_condition_dropout=parse_adapter_condition_dropout,
        parse_adapter_include_mid=parse_adapter_include_mid,
        parse_adapter_zero_init=parse_adapter_zero_init,
        parse_adapter_use_scale_gates=parse_adapter_use_scale_gates,
        parse_adapter_gate_init=parse_adapter_gate_init,
        use_sketch_face_adapter=use_sketch_face_adapter,
        sketch_key=sketch_key,
        sketch_in_channels=sketch_in_channels,
        use_coarse_face_adapter=use_coarse_face_adapter,
        coarse_face_key=coarse_face_key,
        coarse_in_channels=coarse_in_channels,
    )

    training_noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        backbone_signature,
        subfolder="scheduler",
        local_files_only=True,
    )
    sampling_noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        backbone_signature,
        subfolder="scheduler",
        local_files_only=True,
    )

    # LBM Model
    model = LBMModel(
        config,
        denoiser=denoiser,
        training_noise_scheduler=training_noise_scheduler,
        sampling_noise_scheduler=sampling_noise_scheduler,
        vae=vae,
        conditioner=conditioner,
    ).to(torch.bfloat16)

    # ArcFace ID loss 是冻结的识别网络，不参与训练，保持 float32 更稳定
    if getattr(model, "id_loss", None) is not None:
        model.id_loss.float()
        model.id_loss.eval()
        for p in model.id_loss.parameters():
            p.requires_grad = False

    if getattr(model, "ea_dists_loss", None) is not None:
        model.ea_dists_loss.float()
        model.ea_dists_loss.eval()
        for p in model.ea_dists_loss.parameters():
            p.requires_grad = False

    if getattr(model, "local_edge_loss", None) is not None:
        model.local_edge_loss.float()
        model.local_edge_loss.eval()
        for p in model.local_edge_loss.parameters():
            p.requires_grad = False

    if getattr(model, "face_adapter", None) is not None:
        n_adapter = sum(p.numel() for p in model.face_adapter.parameters())
        print(f"[FaceAdapter] parameters: {n_adapter / 1e6:.2f}M")
    return model

class ParseLabelToOneHotMapper:
    """
    Convert single-channel parse label map to one-hot tensor.

    Input:
        batch[key]: PIL Image or tensor with class ids 0~num_classes-1

    Output:
        batch[output_key]: FloatTensor [num_classes, H, W]
    """

    def __init__(
        self,
        key="parse",
        output_key="parse",
        num_classes=19,
        size=(256, 256),
    ):
        self.key = key
        self.output_key = output_key
        self.num_classes = num_classes
        self.size = size

    def __call__(self, batch):
        if self.key not in batch:
            return batch

        parse = batch[self.key]

        # PIL label map: keep integer labels, do not use ToTensor because it divides by 255
        if isinstance(parse, Image.Image):
            parse = parse.convert("L")
            parse = TF.resize(
                parse,
                self.size,
                interpolation=InterpolationMode.NEAREST_EXACT,
            )
            parse_np = np.array(parse).astype(np.int64)
            parse_tensor = torch.from_numpy(parse_np).long()
        elif torch.is_tensor(parse):
            if parse.ndim == 3:
                parse = parse[0]
            parse_tensor = parse.long()
            parse_tensor = F.interpolate(
                parse_tensor[None, None].float(),
                size=self.size,
                mode="nearest",
            )[0, 0].long()
        else:
            raise TypeError(f"Unsupported parse type: {type(parse)}")

        parse_tensor = parse_tensor.clamp(0, self.num_classes - 1)
        onehot = F.one_hot(parse_tensor, num_classes=self.num_classes)
        onehot = onehot.permute(2, 0, 1).float()  # [19,H,W]

        batch[self.output_key] = onehot
        return batch



def get_filter_mappers():
    filters_mappers = [
        KeyFilter(KeyFilterConfig(keys=["__key__", "jpg", "normal_aligned.png", "mask.png", "parse.png"])),
        MapperWrapper(
            [
                KeyRenameMapper(
                    KeyRenameMapperConfig(
                        key_map={
                            "jpg": "image",
                            "normal_aligned.png": "normal",
                            "mask.png": "mask",
                            "parse.png": "parse",
                        }
                    )
                ),
                TorchvisionMapper(
                    TorchvisionMapperConfig(
                        key="image",
                        transforms=["ToTensor", "Resize"],
                        transforms_kwargs=[
                            {},
                            {
                                "size": (256, 256),
                                "interpolation": InterpolationMode.BICUBIC,
                            },
                        ],
                    )
                ),
                TorchvisionMapper(
                    TorchvisionMapperConfig(
                        key="normal",
                        transforms=["ToTensor", "Resize"],
                        transforms_kwargs=[
                            {},
                            {
                                "size": (256, 256),
                                "interpolation": InterpolationMode.BICUBIC,
                            },
                        ],
                    )
                ),
                TorchvisionMapper(
                    TorchvisionMapperConfig(
                        key="mask",
                        transforms=["ToTensor", "Resize", "Normalize"],
                        transforms_kwargs=[
                            {},
                            {
                                "size": (256, 256),
                                "interpolation": InterpolationMode.NEAREST_EXACT,
                            },
                            {"mean": 0.0, "std": 1.0},
                        ],
                    )
                ),
                ParseLabelToOneHotMapper(
                    key="parse",
                    output_key="parse",
                    num_classes=19,
                    size=(256, 256),
                ),
                RescaleMapper(RescaleMapperConfig(key="image")),
                RescaleMapper(RescaleMapperConfig(key="normal")),
            ],
        ),
    ]

    return filters_mappers


def get_data_module(
    train_shards: List[str],
    validation_shards: List[str],
    batch_size: int,
):

    # TRAIN
    train_filters_mappers = get_filter_mappers()

    # unbrace urls
    train_shards_path_or_urls_unbraced = []
    for train_shards_path_or_url in train_shards:
        train_shards_path_or_urls_unbraced.extend(
            braceexpand.braceexpand(train_shards_path_or_url)
        )

    # shuffle shards
    random.shuffle(train_shards_path_or_urls_unbraced)

    # data config
    data_config = DataModuleConfig(
        shards_path_or_urls=train_shards_path_or_urls_unbraced,
        decoder="pil",
        shuffle_before_split_by_node_buffer_size=20,
        shuffle_before_split_by_workers_buffer_size=20,
        shuffle_before_filter_mappers_buffer_size=20,
        shuffle_after_filter_mappers_buffer_size=20,
        per_worker_batch_size=batch_size,
        num_workers=min(10, len(train_shards_path_or_urls_unbraced)),
    )

    train_data_config = data_config

    # VALIDATION
    validation_filters_mappers = get_filter_mappers()

    # unbrace urls
    validation_shards_path_or_urls_unbraced = []
    for validation_shards_path_or_url in validation_shards:
        validation_shards_path_or_urls_unbraced.extend(
            braceexpand.braceexpand(validation_shards_path_or_url)
        )

    data_config = DataModuleConfig(
        shards_path_or_urls=validation_shards_path_or_urls_unbraced,
        decoder="pil",
        shuffle_before_split_by_node_buffer_size=10,
        shuffle_before_split_by_workers_buffer_size=10,
        shuffle_before_filter_mappers_buffer_size=10,
        shuffle_after_filter_mappers_buffer_size=10,
        per_worker_batch_size=batch_size,
        num_workers=min(10, len(train_shards_path_or_urls_unbraced)),
    )

    validation_data_config = data_config

    # data module
    data_module = DataModule(
        train_config=train_data_config,
        train_filters_mappers=train_filters_mappers,
        eval_config=validation_data_config,
        eval_filters_mappers=validation_filters_mappers,
    )

    return data_module


def get_train_regexes(
    train_mode: str,
    use_face_adapter: bool,
    use_vae_skip: bool = False,
):
    """
    Select trainable parameter regexes.

    train_mode:
        backbone_only: train denoiser only
        adapter_only: train face_adapter only
        joint: train denoiser + face_adapter
    """
    valid_modes = {"backbone_only", "adapter_only", "joint"}
    if train_mode not in valid_modes:
        raise ValueError(
            f"Invalid train_mode={train_mode}, must be one of {valid_modes}"
        )

    if train_mode in {"adapter_only", "joint"} and not use_face_adapter:
        raise ValueError(
            f"train_mode={train_mode} requires use_face_adapter=True"
        )

    if train_mode == "backbone_only":
        train_parameters = ["denoiser.*"]

    elif train_mode == "adapter_only":
        train_parameters = ["face_adapter.*"]

    elif train_mode == "joint":
        train_parameters = ["denoiser.*", "face_adapter.*"]

    if use_vae_skip:
        train_parameters.append(r"vae.*skip_conv_.*")

    return train_parameters


def count_trainable_parameters_by_regex(model, train_regexes):
    total = 0
    matched_names = []

    for name, param in model.named_parameters():
        for regex in train_regexes:
            if re.match(regex, name):
                total += param.numel()
                matched_names.append(name)
                break

    return total, matched_names



def load_model_weights_strict_false(pipeline, ckpt_path: str):
    if ckpt_path is None:
        return

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"init_from_checkpoint not found: {ckpt_path}")

    print(f"[InitCheckpoint] Loading model weights from: {ckpt_path}")

    try:
        ckpt = torch.load(
            ckpt_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        ckpt = torch.load(
            ckpt_path,
            map_location="cpu",
        )

    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    current_state = pipeline.state_dict()

    filtered_state = {}
    skipped_missing = []
    skipped_shape = []

    for k, v in state_dict.items():
        if k not in current_state:
            skipped_missing.append(k)
            continue

        if current_state[k].shape != v.shape:
            skipped_shape.append((k, tuple(v.shape), tuple(current_state[k].shape)))
            continue

        filtered_state[k] = v

    missing, unexpected = pipeline.load_state_dict(filtered_state, strict=False)

    face_missing = [k for k in missing if "face_adapter" in k]

    print(f"[InitCheckpoint] loaded tensors: {len(filtered_state)}")
    print(f"[InitCheckpoint] skipped missing keys: {len(skipped_missing)}")
    print(f"[InitCheckpoint] skipped shape mismatch keys: {len(skipped_shape)}")
    print(f"[InitCheckpoint] missing keys after load: {len(missing)}")
    print(f"[InitCheckpoint] unexpected keys after load: {len(unexpected)}")

    if len(face_missing) > 0:
        print(f"[InitCheckpoint] face_adapter newly initialized keys: {len(face_missing)}")

    if len(skipped_shape) > 0:
        print("[InitCheckpoint] shape mismatch examples:")
        for item in skipped_shape[:10]:
            print("  ", item)

class LossLoggerCallback(Callback):
    """Log train/val loss to SwanLab or any Lightning logger."""
    def __init__(self, log_interval: int = 10):
        super().__init__()
        self.log_interval = max(1, int(log_interval))

    @staticmethod
    def _extract_loss(outputs):
        if outputs is None:
            return None

        if torch.is_tensor(outputs):
            return outputs

        if isinstance(outputs, dict):
            for key in ["loss", "train_loss", "val_loss", "loss/train", "loss/val"]:
                if key in outputs and torch.is_tensor(outputs[key]):
                    return outputs[key]

        if isinstance(outputs, (list, tuple)):
            for item in outputs:
                loss = LossLoggerCallback._extract_loss(item)
                if loss is not None:
                    return loss

        return None

    @staticmethod
    def _log_to_all_loggers(trainer, metrics):
        step = trainer.global_step

        if getattr(trainer, "loggers", None):
            for logger in trainer.loggers:
                logger.log_metrics(metrics, step=step)
        elif trainer.logger is not None:
            trainer.logger.log_metrics(metrics, step=step)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.log_interval != 0:
            return

        loss = self._extract_loss(outputs)
        if loss is None:
            return

        loss_value = loss.detach().float().item()
        metrics = {
            "loss/train": loss_value,
            "batch_idx/train": batch_idx,
        }

        # Case 1: TrainingPipeline returns the full dict from LBMModel.forward()
        if isinstance(outputs, dict):
            for k in [
                "latent_recon_loss",
                "pixel_recon_loss",
                "id_recon_loss",
                "adapter_residual_norm",
                "ea_dists_dists_loss",
                "ea_dists_edge_loss",
                "ea_dists_total_loss",
                "local_edge_recon_loss",
                "local_edge_mask_mean",

                "bilbm_latent_s2p",
                "bilbm_latent_p2s",

                "bilbm_pixel_s2p",
                "bilbm_pixel_p2s",
                "bilbm_id_s2p",
                "bilbm_id_p2s",
                "bilbm_local_edge_s2p",
                "bilbm_local_edge_p2s",

                "bilbm_total_s2p",
                "bilbm_total_p2s",
                "bilbm_reverse_loss_weight",

                # Bridge-aware MAAM logs
                "bridge_maam_alpha_mean",
                "bridge_maam_attn_mean",
                "bridge_maam_attn_std",
                "bridge_maam_delta_ratio",
                "bridge_maam_direct_diff_ratio",
            ]:
                if k in outputs and torch.is_tensor(outputs[k]):
                    metrics[f"loss/{k}"] = outputs[k].detach().float().mean().item()

        # Case 2: TrainingPipeline only returns scalar loss.
        # Fallback: read cached values from the underlying LBMModel.
        model = getattr(pl_module, "model", None)
        if model is not None:
            fallback_items = {
                "local_edge_recon_loss": getattr(model, "_last_local_edge_loss", None),
                "local_edge_mask_mean": getattr(model, "_last_local_edge_mask_mean", None),
                "ea_dists_dists_loss": getattr(model, "_last_ea_dists_dists_loss", None),
                "ea_dists_edge_loss": getattr(model, "_last_ea_dists_edge_loss", None),
                "ea_dists_total_loss": getattr(model, "_last_ea_dists_total_loss", None),
            }

            if getattr(model, "denoiser", None) is not None and hasattr(
                    model.denoiser, "get_bridge_maam_log_dict"
            ):
                maam_logs = model.denoiser.get_bridge_maam_log_dict()
                fallback_items.update(maam_logs)

            for k, v in fallback_items.items():
                metric_key = f"loss/{k}"
                if metric_key in metrics:
                    continue
                if torch.is_tensor(v):
                    metrics[metric_key] = v.detach().float().mean().item()

            if hasattr(model, "local_edge_loss_weight"):
                metrics["debug/local_edge_loss_weight"] = float(model.local_edge_loss_weight)

        self._log_to_all_loggers(trainer, metrics)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        loss = self._extract_loss(outputs)
        if loss is None:
            return

        loss_value = loss.detach().float().item()
        self._log_to_all_loggers(
            trainer,
            {
                "loss/val": loss_value,
                "batch_idx/val": batch_idx,
            },
        )

class SwanLabEvalCallback(Callback):
    def __init__(
            self,
            eval_num_steps=(1, 2, 4),
            max_eval_batches=2,
            max_samples_per_batch=2,
            upload_images=True,
            compute_fid=False,
            save_images=False,
            save_size=(200, 250),
            save_dir=None,
            eval_directions: Optional[List[str]] = None,

            external_metrics=False,
            external_metrics_steps=(4,),
            external_metrics_every_n_evals=1,
            external_metrics_device="cpu",
            external_metrics_batch_size=8,
            external_metrics_cuda_visible_devices: Optional[str] = None,
            external_gt_dir=None,
            external_image_metrics=True,
            external_rank_metrics=True,
            external_deepface_home="/root/shuqian/checkpoints",
            external_inception_v3_path="/root/shuqian/checkpoints/inception_v3_google-0cc3c7bd.pth",
            external_metrics_script="tools/calc_external_metrics.py",
            external_metrics_wait_on_fit_end=True,
            external_metrics_wait_timeout=600,
            external_metrics_poll_interval=5,


            external_metrics_directions: Optional[List[str]] = None,
            external_gt_dir_s2p: Optional[str] = None,
            external_gt_dir_p2s: Optional[str] = None,
            external_image_metrics_s2p: bool = True,
            external_rank_metrics_s2p: bool = True,
            external_image_metrics_p2s: bool = True,
            external_rank_metrics_p2s: bool = False,
    ):
        super().__init__()
        self.eval_num_steps = list(eval_num_steps)
        self.max_eval_batches = max_eval_batches
        self.max_samples_per_batch = max_samples_per_batch
        self.cached_val_batches = []
        self.lpips_model = None
        self.upload_images = upload_images
        self.compute_fid = compute_fid
        self.save_images = save_images
        self.save_size = tuple(save_size)  # (width, height)
        self.save_dir = save_dir
        self.eval_directions = eval_directions or ["s2p"]

        self.external_metrics = external_metrics
        self.external_metrics_steps = list(external_metrics_steps)
        self.external_metrics_every_n_evals = max(1, int(external_metrics_every_n_evals))
        self.external_metrics_device = external_metrics_device
        self.external_metrics_batch_size = int(external_metrics_batch_size)
        self.external_gt_dir = external_gt_dir
        self.external_image_metrics = external_image_metrics
        self.external_rank_metrics = external_rank_metrics
        self.external_deepface_home = external_deepface_home
        self.external_inception_v3_path = external_inception_v3_path
        self.external_metrics_script = external_metrics_script
        self.external_metrics_cuda_visible_devices = external_metrics_cuda_visible_devices

        self.external_metrics_directions = external_metrics_directions
        self.external_gt_dir_s2p = external_gt_dir_s2p
        self.external_gt_dir_p2s = external_gt_dir_p2s
        self.external_image_metrics_s2p = external_image_metrics_s2p
        self.external_rank_metrics_s2p = external_rank_metrics_s2p
        self.external_image_metrics_p2s = external_image_metrics_p2s
        self.external_rank_metrics_p2s = external_rank_metrics_p2s

        self.eval_counter = 0
        self.pending_metric_jsons = {}

        self.external_metrics_wait_on_fit_end = external_metrics_wait_on_fit_end
        self.external_metrics_wait_timeout = int(external_metrics_wait_timeout)
        self.external_metrics_poll_interval = int(external_metrics_poll_interval)

    def _get_external_gt_dir_for_direction(self, direction: str):
        if direction == "s2p":
            return self.external_gt_dir_s2p or self.external_gt_dir
        if direction == "p2s":
            return self.external_gt_dir_p2s
        raise ValueError(f"Unknown direction for external metrics: {direction}")

    def _get_external_metric_flags_for_direction(self, direction: str):
        if direction == "s2p":
            return self.external_image_metrics_s2p, self.external_rank_metrics_s2p
        if direction == "p2s":
            return self.external_image_metrics_p2s, self.external_rank_metrics_p2s
        raise ValueError(f"Unknown direction for external metrics: {direction}")

    def setup(self, trainer, pl_module, stage=None):
        if self.lpips_model is None:
            self.lpips_model = lpips.LPIPS(net="vgg")
            self.lpips_model.eval()
            self.lpips_model.to(pl_module.device)

    def on_validation_epoch_start(self, trainer, pl_module):
        self.cached_val_batches = []
        if trainer.is_global_zero:
            print(f"[SwanLabEval] validation start at global_step={trainer.global_step}")

    def _clone_batch_to_cpu(self, batch):
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v[: self.max_samples_per_batch].detach().cpu()
            elif k == "__key__":
                out[k] = list(v[: self.max_samples_per_batch])
        return out

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if len(self.cached_val_batches) >= self.max_eval_batches:
            return
        self.cached_val_batches.append(self._clone_batch_to_cpu(batch))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.is_global_zero:
            self._try_log_finished_external_metrics(trainer)

    def _wait_and_log_finished_external_metrics(self, trainer):
        """
        Wait for all pending external metric json files and log them to SwanLab.

        If external_metrics_wait_timeout > 0:
            wait at most this many seconds.

        If external_metrics_wait_timeout <= 0:
            wait forever until all pending json files are generated and logged.

        This makes external metrics automatically uploaded even after the main
        training loop has finished, as long as this Python process is still alive.
        """
        if not getattr(self, "pending_metric_jsons", None):
            return

        if not self.external_metrics_wait_on_fit_end:
            print(
                "[ExternalMetrics] external_metrics_wait_on_fit_end=False, "
                "only try logging finished jsons once."
            )
            self._try_log_finished_external_metrics(trainer)
            return

        timeout = int(self.external_metrics_wait_timeout)

        if timeout > 0:
            deadline = time.time() + timeout
            print(
                f"[ExternalMetrics] wait for pending external metrics, "
                f"timeout={timeout}s, pending={len(self.pending_metric_jsons)}"
            )
        else:
            deadline = None
            print(
                f"[ExternalMetrics] wait for pending external metrics until all finish, "
                f"timeout=unlimited, pending={len(self.pending_metric_jsons)}"
            )

        last_pending_count = len(self.pending_metric_jsons)

        while self.pending_metric_jsons:
            # Try to read and log json files that have already been generated.
            self._try_log_finished_external_metrics(trainer)

            if not self.pending_metric_jsons:
                print("[ExternalMetrics] all pending external metrics have been logged.")
                break

            # Timeout only works when external_metrics_wait_timeout > 0.
            if deadline is not None and time.time() >= deadline:
                print(
                    "[ExternalMetrics] wait timeout reached. "
                    f"Unlogged pending jsons: {list(self.pending_metric_jsons.keys())}"
                )
                break

            current_pending_count = len(self.pending_metric_jsons)

            if current_pending_count != last_pending_count:
                print(
                    f"[ExternalMetrics] pending reduced: "
                    f"{last_pending_count} -> {current_pending_count}"
                )
                last_pending_count = current_pending_count

            print(
                f"[ExternalMetrics] waiting for {len(self.pending_metric_jsons)} "
                f"external metric task(s) to finish..."
            )

            time.sleep(self.external_metrics_poll_interval)

        # One final try before leaving.
        self._try_log_finished_external_metrics(trainer)


    def on_fit_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            self._wait_and_log_finished_external_metrics(trainer)
            self._try_log_finished_external_metrics(trainer)

    @staticmethod
    def _to_01(x):
        # 输入通常是 [-1, 1]，转到 [0, 1]
        return (x.clamp(-1, 1) + 1.0) / 2.0

    @staticmethod
    def _to_uint8_np(img_tensor):
        # [C,H,W] -> [H,W,C] uint8
        img = img_tensor.detach().cpu().clamp(0, 1)
        img = (img * 255.0).byte().permute(1, 2, 0).numpy()
        return img

    def _save_generated_images(
            self,
            trainer,
            pred,
            step_num,
            batch_idx,
            keys=None,
            direction: str = "s2p",
    ):
        if not self.save_images:
            return None

        root_dir = self.save_dir
        if root_dir is None:
            root_dir = os.path.join(trainer.default_root_dir, "eval_images")

        gen_dir = os.path.join(
            root_dir,
            f"global_step_{trainer.global_step:08d}",
            f"step_{step_num}",
            direction,
            "generated",
        )
        os.makedirs(gen_dir, exist_ok=True)

        for i in range(pred.shape[0]):
            if keys is not None:
                stem = str(keys[i])
            else:
                stem = str(batch_idx * self.max_samples_per_batch + i + 1)

            filename = f"{stem}.jpg"

            img_np = self._to_uint8_np(pred[i])
            img = Image.fromarray(img_np)
            img = img.resize(self.save_size, Image.LANCZOS)

            save_path = os.path.join(gen_dir, filename)
            img.save(save_path, quality=100, subsampling=0)

        return gen_dir

    def _log_metrics(self, trainer, metrics: dict, step=None):
        if step is None:
            step = trainer.global_step

        if getattr(trainer, "loggers", None):
            for logger in trainer.loggers:
                logger.log_metrics(metrics, step=step)
        elif trainer.logger is not None:
            trainer.logger.log_metrics(metrics, step=step)

    def _try_log_finished_external_metrics(self, trainer):
        if not self.pending_metric_jsons:
            return

        finished = []

        for json_path, metric_step in list(self.pending_metric_jsons.items()):
            if not os.path.exists(json_path):
                continue

            try:
                with open(json_path, "r") as f:
                    raw_metrics = json.load(f)

                # ------------------------------------------------------------
                # Infer direction from json path.
                # Current _launch_external_metrics saves json as:
                #   .../external_metrics/s2p/metrics_global_step_xxx_s2p_step_1.json
                #   .../external_metrics/p2s/metrics_global_step_xxx_p2s_step_1.json
                # ------------------------------------------------------------
                direction = None
                norm_path = os.path.normpath(json_path)
                path_parts = norm_path.split(os.sep)

                if "external_metrics" in path_parts:
                    idx = path_parts.index("external_metrics")
                    if idx + 1 < len(path_parts):
                        maybe_direction = path_parts[idx + 1]
                        if maybe_direction in ["s2p", "p2s"]:
                            direction = maybe_direction

                if direction is None:
                    base = os.path.basename(json_path)
                    if "_s2p_" in base or base.startswith("s2p_"):
                        direction = "s2p"
                    elif "_p2s_" in base or base.startswith("p2s_"):
                        direction = "p2s"

                # Fallback: old behavior
                if direction is None:
                    direction = "unknown"

                # ------------------------------------------------------------
                # Keep numeric metrics only and add direction prefix.
                #
                # Example:
                #   external/fid_step1 -> external/s2p/fid_step1
                #   external/rank1_step1 -> external/s2p/rank1_step1
                #
                # If a key is already prefixed, keep it unchanged.
                # ------------------------------------------------------------
                metrics = {}
                for k, v in raw_metrics.items():
                    if not isinstance(v, (int, float)):
                        continue

                    v = float(v)

                    if direction in ["s2p", "p2s"]:
                        prefix = f"external/{direction}/"

                        if k.startswith(prefix):
                            new_key = k
                        elif k.startswith("external/"):
                            new_key = k.replace("external/", prefix, 1)
                        else:
                            new_key = prefix + k
                    else:
                        new_key = k

                    metrics[new_key] = v

                if len(metrics) > 0:
                    self._log_metrics(trainer, metrics, step=metric_step)

                    try:
                        swanlab.log(metrics, step=metric_step)
                    except Exception as e:
                        print(f"[ExternalMetrics] swanlab.log failed: {repr(e)}")

                    print(
                        f"[ExternalMetrics] logged direction={direction} "
                        f"at step={metric_step}: {metrics}"
                    )

                finished.append(json_path)

            except Exception as e:
                print(f"[ExternalMetrics] failed to read {json_path}: {repr(e)}")

        for p in finished:
            self.pending_metric_jsons.pop(p, None)

    def _launch_external_metrics(
            self,
            trainer,
            gen_dir,
            step_num,
            direction="s2p",
            gt_dir=None,
            do_image_metrics=None,
            do_rank_metrics=None,
    ):
        if not self.external_metrics:
            return

        if step_num not in self.external_metrics_steps:
            return

        # ------------------------------------------------------------
        # Direction-aware gt_dir selection
        # ------------------------------------------------------------
        if gt_dir is None:
            if direction == "s2p":
                # s2p: generated photo should compare with real photo
                gt_dir = getattr(self, "external_gt_dir_s2p", None)
                if gt_dir is None:
                    gt_dir = self.external_gt_dir
            elif direction == "p2s":
                # p2s: generated sketch should compare with real sketch
                gt_dir = getattr(self, "external_gt_dir_p2s", None)
            else:
                print(f"[ExternalMetrics] unknown direction={direction}, skip.")
                return

        if gt_dir is None:
            print(
                f"[ExternalMetrics] gt_dir is None, skip. "
                f"direction={direction}, step={step_num}"
            )
            return

        if not os.path.isdir(gt_dir):
            print(
                f"[ExternalMetrics] invalid gt_dir, skip. "
                f"direction={direction}, step={step_num}, gt_dir={gt_dir}"
            )
            return

        if gen_dir is None or not os.path.isdir(gen_dir):
            print(
                f"[ExternalMetrics] invalid gen_dir, skip. "
                f"direction={direction}, step={step_num}, gen_dir={gen_dir}"
            )
            return

        # ------------------------------------------------------------
        # Direction-aware metric flags
        # ------------------------------------------------------------
        if do_image_metrics is None:
            if direction == "s2p":
                do_image_metrics = getattr(
                    self,
                    "external_image_metrics_s2p",
                    self.external_image_metrics,
                )
            elif direction == "p2s":
                do_image_metrics = getattr(
                    self,
                    "external_image_metrics_p2s",
                    self.external_image_metrics,
                )
            else:
                do_image_metrics = self.external_image_metrics

        if do_rank_metrics is None:
            if direction == "s2p":
                do_rank_metrics = getattr(
                    self,
                    "external_rank_metrics_s2p",
                    self.external_rank_metrics,
                )
            elif direction == "p2s":
                do_rank_metrics = getattr(
                    self,
                    "external_rank_metrics_p2s",
                    False,
                )
            else:
                do_rank_metrics = self.external_rank_metrics

        if not do_image_metrics and not do_rank_metrics:
            print(
                f"[ExternalMetrics] both image/rank metrics disabled, skip. "
                f"direction={direction}, step={step_num}"
            )
            return

        root_dir = self.save_dir
        if root_dir is None:
            root_dir = os.path.join(trainer.default_root_dir, "eval_images")

        # Separate direction folders to avoid json/log overwrite
        metrics_dir = os.path.join(root_dir, "external_metrics", direction)
        os.makedirs(metrics_dir, exist_ok=True)

        out_json = os.path.join(
            metrics_dir,
            f"metrics_global_step_{trainer.global_step:08d}_{direction}_step_{step_num}.json",
        )

        # 只避免同一个 json 重复启动，不阻止 step1/step4 或 s2p/p2s 同时启动
        if out_json in self.pending_metric_jsons:
            print(f"[ExternalMetrics] task already pending, skip: {out_json}")
            return

        # ------------------------------------------------------------
        # Decide device and visible CUDA devices for the external metric
        # child process.
        #
        # Example:
        #   external_metrics_device = "cuda:0"
        #   external_metrics_cuda_visible_devices = "3"
        #
        # Then the child process only sees physical GPU 3, and inside the
        # child process it is mapped to cuda:0.
        # ------------------------------------------------------------
        metric_device = str(self.external_metrics_device)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["OMP_NUM_THREADS"] = "2"
        env["MKL_NUM_THREADS"] = "2"
        env["OPENBLAS_NUM_THREADS"] = "2"
        env["NUMEXPR_NUM_THREADS"] = "2"
        env["VECLIB_MAXIMUM_THREADS"] = "2"
        env["TORCH_NUM_THREADS"] = "2"

        if metric_device.lower() == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
            metric_device_for_child = "cpu"
        else:
            visible_devices = getattr(
                self,
                "external_metrics_cuda_visible_devices",
                None,
            )

            if visible_devices is not None and str(visible_devices).strip() != "":
                env["CUDA_VISIBLE_DEVICES"] = str(visible_devices)

                # After CUDA_VISIBLE_DEVICES is set, the selected physical GPU(s)
                # are remapped inside the child process. Therefore, use cuda:0.
                if metric_device.startswith("cuda"):
                    metric_device_for_child = "cuda:0"
                else:
                    metric_device_for_child = metric_device
            else:
                # Use the same CUDA_VISIBLE_DEVICES environment as the training process.
                metric_device_for_child = metric_device

        cmd = [
            sys.executable,
            "-u",
            self.external_metrics_script,
            "--gt_dir", str(gt_dir),
            "--gen_dir", str(gen_dir),
            "--out_json", str(out_json),
            "--device", str(metric_device_for_child),
            "--batch_size", str(self.external_metrics_batch_size),
            "--deepface_home", str(self.external_deepface_home),
            "--local_inception_v3_path", str(self.external_inception_v3_path),
            "--step_num", str(step_num),
        ]

        if do_image_metrics:
            cmd.append("--do_image_metrics")

        if do_rank_metrics:
            cmd.append("--do_rank_metrics")

        log_path = out_json + ".log"

        print(
            f"[ExternalMetrics] launch direction={direction}, step={step_num}, "
            f"image_metrics={do_image_metrics}, rank_metrics={do_rank_metrics}: "
            f"{' '.join(cmd)}"
        )
        print(f"[ExternalMetrics] gt_dir direction={direction}, step={step_num}: {gt_dir}")
        print(f"[ExternalMetrics] gen_dir direction={direction}, step={step_num}: {gen_dir}")
        print(f"[ExternalMetrics] out_json direction={direction}, step={step_num}: {out_json}")
        print(f"[ExternalMetrics] log direction={direction}, step={step_num}: {log_path}")
        print(
            f"[ExternalMetrics] device={metric_device_for_child}, "
            f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', os.environ.get('CUDA_VISIBLE_DEVICES', None))}"
        )

        with open(log_path, "w") as log_f:
            subprocess.Popen(
                cmd,
                cwd=os.getcwd(),
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
            )

        self.pending_metric_jsons[out_json] = trainer.global_step

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        if len(self.cached_val_batches) == 0:
            print(f"[SwanLabEval] no cached validation batch at global_step={trainer.global_step}")
            return
        print(
            f"[SwanLabEval] running eval at global_step={trainer.global_step}, "
            f"cached_batches={len(self.cached_val_batches)}"
        )

        device = pl_module.device
        self.lpips_model = self.lpips_model.to(device)
        self.lpips_model.eval()

        metric_buffers = {}
        saved_gen_dirs = {}

        fid_metrics = None
        if self.compute_fid:
            fid_metrics = {
                step_num: FrechetInceptionDistance(
                    feature=2048,
                    normalize=True,
                ).to(device)
                for step_num in self.eval_num_steps
            }

        was_training = pl_module.training
        pl_module.eval()

        for batch_idx, batch_cpu in enumerate(self.cached_val_batches):
            batch = {}
            for k, v in batch_cpu.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device=device, dtype=torch.bfloat16)
                elif k == "__key__":
                    batch[k] = v

            keys = batch.get("__key__", None)

            for direction in self.eval_directions:
                if direction == "s2p":
                    src_pixels = batch[pl_module.model.source_key]
                    gt_pixels = batch[pl_module.model.target_key]
                elif direction == "p2s":
                    src_pixels = batch[pl_module.model.target_key]
                    gt_pixels = batch[pl_module.model.source_key]
                else:
                    raise ValueError(f"Unknown eval direction: {direction}")

                # 当前第一版 eval_directions=["s2p"]，所以这里只会跑 sketch->photo。
                # 后面打开 p2s 时，这里会自动改成 photo->sketch。
                src = self._to_01(src_pixels.float())
                gt = self._to_01(gt_pixels.float())

                for step_num in self.eval_num_steps:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        src_for_vae = src_pixels.to(device=device, dtype=torch.bfloat16)

                        if pl_module.model.vae is not None:
                            # 保持和 LBMModel.log_samples 原逻辑一致：source resize 到 target 空间大小
                            src_for_vae = torch.nn.functional.interpolate(
                                src_for_vae,
                                size=gt_pixels.shape[2:],
                                mode="bilinear",
                                align_corners=False,
                            ).to(dtype=pl_module.model.dtype)

                            z_src = pl_module.model.vae.encode(src_for_vae)
                        else:
                            z_src = src_for_vae

                        pred_raw = pl_module.model.sample(
                            z_src,
                            num_steps=step_num,
                            conditioner_inputs=batch,
                            max_samples=src_pixels.shape[0],
                            verbose=False,
                            direction=direction,
                        )

                    pred = self._to_01(pred_raw.float())

                    gen_dir = self._save_generated_images(
                        trainer=trainer,
                        pred=pred,
                        step_num=step_num,
                        batch_idx=batch_idx,
                        keys=keys,
                        direction=direction,
                    )

                    if gen_dir is not None:
                        saved_gen_dirs[(direction, step_num)] = gen_dir

                    # ---------- 计算指标 ----------
                    # 第一版 eval_directions=["s2p"]，所以指标仍然只针对 sketch->photo。
                    # 如果后面打开 p2s，这里会给 p2s 单独记录 eval/p2s/*。
                    ssim_val = structural_similarity_index_measure(
                        pred, gt, data_range=1.0
                    ).item()

                    pred_lp = pred * 2.0 - 1.0
                    gt_lp = gt * 2.0 - 1.0
                    lpips_val = self.lpips_model(pred_lp, gt_lp).mean().item()

                    metric_buffers.setdefault(
                        f"eval/{direction}/ssim_step{step_num}", []
                    ).append(ssim_val)
                    metric_buffers.setdefault(
                        f"eval/{direction}/lpips_step{step_num}", []
                    ).append(lpips_val)

                    if fid_metrics is not None:
                        # 第一版建议 compute_fid=False。
                        # 如果后面要支持 p2s，最好给不同 direction 分开建 FID metric。
                        if direction == "s2p":
                            fid_metrics[step_num].update(gt.float(), real=True)
                            fid_metrics[step_num].update(pred.float(), real=False)

                    if self.upload_images:
                        triplets = []
                        for i in range(src.shape[0]):
                            triplet = torch.cat([src[i], pred[i], gt[i]], dim=2)
                            triplets.append(triplet)

                        grid = make_grid(triplets, nrow=1)
                        grid_np = self._to_uint8_np(grid)

                        swanlab.log(
                            {
                                f"eval/{direction}/images_step{step_num}_batch{batch_idx}": swanlab.Image(
                                    grid_np,
                                    caption=(
                                        f"direction={direction} | step={step_num} | "
                                        f"batch={batch_idx} | input | generated | gt"
                                    ),
                                )
                            },
                            step=trainer.global_step,
                        )

        # ---------- 记录平均指标 ----------
        final_metrics = {}
        for k, vals in metric_buffers.items():
            if len(vals) > 0:
                final_metrics[k] = float(np.mean(vals))

        if fid_metrics is not None:
            for step_num, fid_metric in fid_metrics.items():
                try:
                    fid_value = fid_metric.compute().detach().float().item()
                    final_metrics[f"eval/fid_step{step_num}"] = fid_value
                except Exception as e:
                    logging.warning(f"FID compute failed for step {step_num}: {e}")

        self.eval_counter += 1

        if (
                self.external_metrics
                and self.eval_counter % self.external_metrics_every_n_evals == 0
        ):
            # External metrics are kept S2P-only in this first version.
            # P2S needs a different GT directory and should not use DeepFace/Rank directly.
            if self.external_metrics:
                metrics_directions = self.external_metrics_directions or ["s2p"]

                for direction in metrics_directions:
                    gt_dir = self._get_external_gt_dir_for_direction(direction)
                    do_image_metrics, do_rank_metrics = self._get_external_metric_flags_for_direction(direction)

                    if gt_dir is None:
                        print(f"[ExternalMetrics] skip direction={direction}: gt_dir is None")
                        continue

                    if not do_image_metrics and not do_rank_metrics:
                        print(f"[ExternalMetrics] skip direction={direction}: both image/rank metrics disabled")
                        continue

                    for ext_step in self.external_metrics_steps:
                        gen_dir = saved_gen_dirs.get((direction, ext_step), None)

                        if gen_dir is None:
                            print(
                                f"[ExternalMetrics] skip direction={direction}, step={ext_step}: "
                                f"generated dir not found in saved_gen_dirs"
                            )
                            continue

                        metrics = self._launch_external_metrics(
                            trainer=trainer,
                            gen_dir=gen_dir,
                            step_num=ext_step,
                            direction=direction,
                            gt_dir=gt_dir,
                            do_image_metrics=do_image_metrics,
                            do_rank_metrics=do_rank_metrics,
                        )

                        if metrics is not None and len(metrics) > 0:
                            final_metrics.update(metrics)

        if len(final_metrics) > 0:
            self._log_metrics(trainer, final_metrics)
            # swanlab.log(final_metrics, step=trainer.global_step)

        if was_training:
            pl_module.train()




def main(
    train_shards: List[str] = ["pipe:cat path/to/train/shards"],
    validation_shards: List[str] = ["pipe:cat path/to/validation/shards"],
    backbone_signature: str = "stabilityai/stable-diffusion-xl-base-1.0",
    vae_num_channels: int = 4,
    unet_input_channels: int = 4,
    source_key: str = "image",
    target_key: str = "normal",
    mask_key: str = "mask",
    wandb_project: str = "lbm-surface",
    batch_size: int = 8,
    num_steps: List[int] = [1, 2, 4],
    learning_rate: float = 5e-5,
    learning_rate_scheduler: str = None,
    learning_rate_scheduler_kwargs: dict = {},
    optimizer: str = "AdamW",
    optimizer_kwargs: dict = {},
    timestep_sampling: str = "uniform",
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    pixel_loss_type: str = "lpips",
    latent_loss_type: str = "l2",
    latent_loss_weight: float = 1.0,
    pixel_loss_weight: float = 0.0,
    ea_dists_edge_weight: float = 1.0,
    ea_dists_use_edge: bool = True,
    ea_dists_edge_to_rgb: bool = True,
    ea_dists_edge_normalize: bool = True,
    ea_dists_resize_to: int = None,
    local_edge_loss_weight: float = 0.0,
    local_edge_parts: List[str] = None,
    local_edge_dilate_kernel: int = 7,
    local_edge_exclude_labels: List[int] = None,
    local_edge_exclude_dilate_kernel: int = 7,
    # bidirectional LBM
    bidirectional: bool = False,
    bidirectional_mode: str = "none",
    direction_aware: bool = False,
    num_directions: int = 2,
    direction_embed_init: float = 0.0,
    reverse_loss_weight: float = 0.5,
    reverse_use_pixel_loss: bool = False,
    reverse_pixel_loss_type: str = None,
    reverse_pixel_loss_weight: float = None,
    reverse_use_id_loss: bool = False,
    reverse_use_local_edge_loss: bool = False,
    # bridge-aware MAAM skip refinement
    use_bridge_maam: bool = False,
    bridge_maam_mode: str = "residual",
    bridge_maam_levels: List[str] = None,
    bridge_maam_attn_type: str = "scsa",
    bridge_maam_alpha_init: float = 0.01,
    bridge_maam_attn_bias_init: float = 2.0,
    bridge_maam_use_timestep: bool = True,
    bridge_maam_zero_init_timestep: bool = True,
    bridge_maam_scsa_groups: int = 4,
    bridge_maam_scsa_kernels: List[int] = None,
    bridge_maam_scsa_pool_size: int = 7,
    # vae skip
    use_vae_skip: bool = False,
    vae_skip_zero_init: bool = True,
    vae_skip_gamma: float = 1.0,
    id_loss_weight: float = 0.0,
    id_loss_model_path: str = None,
    id_loss_crop: bool = True,
    selected_timesteps: List[float] = None,
    prob: List[float] = None,
    conditioning_images_keys: Optional[List[str]] = [],
    conditioning_masks_keys: Optional[List[str]] = [],
    config_yaml: dict = None,
    save_ckpt_path: str = "./checkpoints",
    log_interval: int = 100,
    resume_from_checkpoint: bool = True,
    init_from_checkpoint: str = None,
    train_mode: str = "backbone_only",
    max_epochs: int = 100,
    bridge_noise_sigma: float = 0.005,
    save_interval: int = 1000,
    path_config: str = None,
    val_check_interval: int = 1000,
    limit_val_batches: int = 2,
    eval_num_steps: List[int] = [1, 2, 4],
    eval_max_batches: int = 2,
    eval_max_samples: int = 2,
    eval_upload_images: bool = True,
    eval_compute_fid: bool = False,
    eval_save_images: bool = False,
    eval_save_size: List[int] = [200, 250],
    eval_save_dir: str = None,
    eval_directions: List[str] = None,
    eval_save_p2s_images: bool = False,
    eval_external_metrics: bool = False,
    eval_external_metrics_steps: List[int] = [4],
    eval_external_metrics_every_n_evals: int = 1,
    eval_external_metrics_device: str = "cpu",
    eval_external_metrics_batch_size: int = 8,
    eval_external_metrics_cuda_visible_devices: str = None,


    # old global fields, kept for compatibility
    eval_gt_dir: str = None,
    eval_external_image_metrics: bool = True,
    eval_external_rank_metrics: bool = True,

    # new direction-aware external metrics fields
    eval_external_metrics_directions: List[str] = None,
    eval_gt_dir_s2p: str = None,
    eval_gt_dir_p2s: str = None,
    eval_external_image_metrics_s2p: bool = True,
    eval_external_rank_metrics_s2p: bool = True,
    eval_external_image_metrics_p2s: bool = True,
    eval_external_rank_metrics_p2s: bool = False,
    eval_external_deepface_home: str = "/root/shuqian/checkpoints",
    eval_external_inception_v3_path: str = "/root/shuqian/checkpoints/inception_v3_google-0cc3c7bd.pth",
    eval_external_metrics_script: str = "tools/calc_external_metrics.py",
    eval_external_metrics_wait_on_fit_end: bool = True,
    eval_external_metrics_wait_timeout: int = 600,
    eval_external_metrics_poll_interval: int = 5,
    # face adapter
    use_face_adapter: bool = False,
    parse_key: str = "parse",
    parse_num_classes: int = 19,
    parse_adapter_scale: float = 1.0,
    parse_adapter_condition_dropout: float = 0.0,
    parse_adapter_include_mid: bool = True,
    parse_adapter_zero_init: bool = True,
    parse_adapter_use_scale_gates: bool = True,
    parse_adapter_gate_init: float = 1.0,
    use_sketch_face_adapter: bool = False,
    sketch_key: str = None,
    sketch_in_channels: int = 3,
    use_coarse_face_adapter: bool = False,
    coarse_face_key: str = None,
    coarse_in_channels: int = 3,
):
    model = get_model(
        backbone_signature=backbone_signature,
        vae_num_channels=vae_num_channels,
        unet_input_channels=unet_input_channels,
        source_key=source_key,
        target_key=target_key,
        mask_key=mask_key,
        timestep_sampling=timestep_sampling,
        logit_mean=logit_mean,
        logit_std=logit_std,
        pixel_loss_type=pixel_loss_type,
        latent_loss_type=latent_loss_type,
        latent_loss_weight=latent_loss_weight,
        pixel_loss_weight=pixel_loss_weight,
        ea_dists_edge_weight=ea_dists_edge_weight,
        ea_dists_use_edge=ea_dists_use_edge,
        ea_dists_edge_to_rgb=ea_dists_edge_to_rgb,
        ea_dists_edge_normalize=ea_dists_edge_normalize,
        ea_dists_resize_to=ea_dists_resize_to,
        local_edge_loss_weight=local_edge_loss_weight,
        local_edge_parts=local_edge_parts,
        local_edge_dilate_kernel=local_edge_dilate_kernel,
        local_edge_exclude_labels=local_edge_exclude_labels,
        local_edge_exclude_dilate_kernel=local_edge_exclude_dilate_kernel,
        use_bridge_maam=use_bridge_maam,
        bridge_maam_mode=bridge_maam_mode,
        bridge_maam_levels=bridge_maam_levels,
        bridge_maam_attn_type=bridge_maam_attn_type,
        bridge_maam_alpha_init=bridge_maam_alpha_init,
        bridge_maam_attn_bias_init=bridge_maam_attn_bias_init,
        bridge_maam_use_timestep=bridge_maam_use_timestep,
        bridge_maam_zero_init_timestep=bridge_maam_zero_init_timestep,
        bridge_maam_scsa_groups=bridge_maam_scsa_groups,
        bridge_maam_scsa_kernels=bridge_maam_scsa_kernels,
        bridge_maam_scsa_pool_size=bridge_maam_scsa_pool_size,
        use_vae_skip=use_vae_skip,
        vae_skip_zero_init=vae_skip_zero_init,
        vae_skip_gamma=vae_skip_gamma,
        id_loss_weight=id_loss_weight,
        id_loss_model_path=id_loss_model_path,
        id_loss_crop=id_loss_crop,
        selected_timesteps=selected_timesteps,
        prob=prob,
        conditioning_images_keys=conditioning_images_keys,
        conditioning_masks_keys=conditioning_masks_keys,
        bridge_noise_sigma=bridge_noise_sigma,
        bidirectional=bidirectional,
        bidirectional_mode=bidirectional_mode,
        direction_aware=direction_aware,
        num_directions=num_directions,
        direction_embed_init=direction_embed_init,
        reverse_loss_weight=reverse_loss_weight,
        reverse_use_pixel_loss=reverse_use_pixel_loss,
        reverse_pixel_loss_type=reverse_pixel_loss_type,
        reverse_pixel_loss_weight=reverse_pixel_loss_weight,
        reverse_use_id_loss=reverse_use_id_loss,
        reverse_use_local_edge_loss=reverse_use_local_edge_loss,
        eval_directions=eval_directions,
        eval_save_p2s_images=eval_save_p2s_images,
        #face adapter
        use_face_adapter=use_face_adapter,
        parse_key=parse_key,
        parse_num_classes=parse_num_classes,
        parse_adapter_scale=parse_adapter_scale,
        parse_adapter_condition_dropout=parse_adapter_condition_dropout,
        parse_adapter_include_mid=parse_adapter_include_mid,
        parse_adapter_zero_init=parse_adapter_zero_init,
        parse_adapter_use_scale_gates=parse_adapter_use_scale_gates,
        parse_adapter_gate_init=parse_adapter_gate_init,
        use_sketch_face_adapter=use_sketch_face_adapter,
        sketch_key=sketch_key,
        sketch_in_channels=sketch_in_channels,
        use_coarse_face_adapter=use_coarse_face_adapter,
        coarse_face_key=coarse_face_key,
        coarse_in_channels=coarse_in_channels,
    )

    data_module = get_data_module(
        train_shards=train_shards,
        validation_shards=validation_shards,
        batch_size=batch_size,
    )

    train_parameters = get_train_regexes(
        train_mode=train_mode,
        use_face_adapter=use_face_adapter,
        use_vae_skip=use_vae_skip,
    )

    print(f"[TrainMode] train_mode = {train_mode}")
    print(f"[TrainMode] trainable regexes = {train_parameters}")

    n_trainable, matched_names = count_trainable_parameters_by_regex(
        model,
        train_parameters,
    )
    print(f"[TrainMode] trainable parameters = {n_trainable / 1e6:.2f}M")
    print("[TrainMode] first 20 trainable parameter names:")
    for name in matched_names[:20]:
        print("   ", name)

    # Training Config
    training_config = TrainingConfig(
        learning_rate=learning_rate,
        lr_scheduler_name=learning_rate_scheduler,
        lr_scheduler_kwargs=learning_rate_scheduler_kwargs,
        log_keys=["image", "normal", "mask"],
        trainable_params=train_parameters,
        optimizer_name=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        log_samples_model_kwargs={
            "input_shape": None,
            "num_steps": num_steps,
        },
    )
    if (
        os.path.exists(save_ckpt_path)
        and resume_from_checkpoint
        and "last.ckpt" in os.listdir(save_ckpt_path)
    ):
        start_ckpt = f"{save_ckpt_path}/last.ckpt"
        print(f"Resuming from checkpoint: {start_ckpt}")

    else:
        start_ckpt = None

    pipeline = TrainingPipeline(model=model, pipeline_config=training_config)

    if init_from_checkpoint is not None:
        load_model_weights_strict_false(pipeline, init_from_checkpoint)

    pipeline.save_hyperparameters(
        {
            f"embedder_{i}": embedder.config.to_dict()
            for i, embedder in enumerate(model.conditioner.conditioners)
        }
    )

    pipeline.save_hyperparameters(
        {
            "denoiser": model.denoiser.config,
            "vae": model.vae.config.to_dict(),
            "config_yaml": config_yaml,
            "training": training_config.to_dict(),
            "training_noise_scheduler": model.training_noise_scheduler.config,
            "sampling_noise_scheduler": model.sampling_noise_scheduler.config,
        }
    )

    training_signature = (
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        + "-LBM-Surface"
        + f"{os.environ['SLURM_JOB_ID']}"
        + f"_{os.environ.get('SLURM_ARRAY_TASK_ID', 0)}"
    )
    dir_path = f"{save_ckpt_path}/logs/{training_signature}"
    if os.environ["SLURM_PROCID"] == "0":
        os.makedirs(dir_path, exist_ok=True)
        if path_config is not None:
            shutil.copy(path_config, f"{save_ckpt_path}/config.yaml")
    run_name = training_signature

    # Ignore parameters unused during training
    ignore_states = []
    active_train_regex = train_parameters

    for name, param in pipeline.model.named_parameters():
        ignore = True
        for regex in active_train_regex:
            pattern = re.compile(regex)
            if re.match(pattern, name):
                ignore = False
                break

        if ignore:
            ignore_states.append(param)

    # FSDP Strategy
    strategy = FSDPStrategy(
        auto_wrap_policy=ModuleWrapPolicy(
            [
                UNet2DConditionModel,
                BasicTransformerBlock,
                ResnetBlock2D,
                # torch.nn.Conv2d,  # 不要包普通 Conv2d，避免把 ArcFace IDLoss 里的卷积也包进去
            ]
        ),
        activation_checkpointing_policy=ModuleWrapPolicy(
            [
                BasicTransformerBlock,
                ResnetBlock2D,
            ]
        ),
        sharding_strategy="SHARD_GRAD_OP",
        ignored_states=ignore_states,
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=int(os.environ["SLURM_NPROCS"]) // int(os.environ["SLURM_NNODES"]),
        num_nodes=int(os.environ["SLURM_NNODES"]),
        strategy=strategy,
        default_root_dir="logs",
        logger=SwanLabLogger(
            project=wandb_project,
            experiment_name=run_name,
            logdir=save_ckpt_path,
        ),
        callbacks=[
            LossLoggerCallback(log_interval=log_interval),
            SwanLabEvalCallback(
                eval_num_steps=eval_num_steps,
                max_eval_batches=eval_max_batches,
                max_samples_per_batch=eval_max_samples,
                upload_images=eval_upload_images,
                compute_fid=eval_compute_fid,
                save_images=eval_save_images,
                save_size=eval_save_size,
                save_dir=eval_save_dir if eval_save_dir is not None else os.path.join(save_ckpt_path, "eval_images"),
                eval_directions=eval_directions,

                external_metrics=eval_external_metrics,
                external_metrics_steps=eval_external_metrics_steps,
                external_metrics_every_n_evals=eval_external_metrics_every_n_evals,
                external_metrics_device=eval_external_metrics_device,
                external_metrics_batch_size=eval_external_metrics_batch_size,

                # old global fields
                external_gt_dir=eval_gt_dir,
                external_image_metrics=eval_external_image_metrics,
                external_rank_metrics=eval_external_rank_metrics,

                # new direction-aware fields
                external_metrics_directions=eval_external_metrics_directions,
                external_gt_dir_s2p=eval_gt_dir_s2p,
                external_gt_dir_p2s=eval_gt_dir_p2s,
                external_image_metrics_s2p=eval_external_image_metrics_s2p,
                external_rank_metrics_s2p=eval_external_rank_metrics_s2p,
                external_image_metrics_p2s=eval_external_image_metrics_p2s,
                external_rank_metrics_p2s=eval_external_rank_metrics_p2s,

                external_deepface_home=eval_external_deepface_home,
                external_inception_v3_path=eval_external_inception_v3_path,
                external_metrics_script=eval_external_metrics_script,
                external_metrics_wait_on_fit_end=eval_external_metrics_wait_on_fit_end,
                external_metrics_wait_timeout=eval_external_metrics_wait_timeout,
                external_metrics_poll_interval=eval_external_metrics_poll_interval,
                external_metrics_cuda_visible_devices=eval_external_metrics_cuda_visible_devices,
            ),
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                dirpath=save_ckpt_path,
                filename="epoch={epoch:03d}-step={step}",
                every_n_train_steps=save_interval,
                # save_top_k=-1,
                save_last=True,
            )
        ],
        num_sanity_val_steps=0,
        precision="bf16-mixed",
        limit_val_batches=limit_val_batches,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=None,
        max_epochs=max_epochs,
        log_every_n_steps=log_interval,
    )

    trainer.fit(pipeline, data_module, ckpt_path=start_ckpt)


def main_from_config(path_config: str = None):
    with open(path_config, "r") as file:
        config = yaml.safe_load(file)
    logging.info(
        f"Running main with config: {yaml.dump(config, default_flow_style=False)}"
    )
    main(**config, config_yaml=config, path_config=path_config)


if __name__ == "__main__":
    fire.Fire(main_from_config)
