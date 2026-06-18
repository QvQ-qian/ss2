# src/lbm/models/unets/unet.py

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from diffusers.models import UNet2DConditionModel, UNet2DModel

from ..bridge_maam import BridgeAwareMAAMSkipRefiner


class DiffusersUNet2DWrapper(UNet2DModel):
    """
    Wrapper for the unconditional UNet2DModel from diffusers.
    """

    def __init__(self, *args, **kwargs):
        UNet2DModel.__init__(self, *args, **kwargs)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        conditioning: Dict[str, torch.Tensor] = None,
        *args,
        **kwargs,
    ):
        if conditioning is not None:
            class_labels = conditioning["cond"].get("vector", None)
            concat = conditioning["cond"].get("concat", None)
        else:
            class_labels = None
            concat = None

        if concat is not None:
            sample = torch.cat([sample, concat], dim=1)

        return (
            super()
            .forward(
                sample=sample,
                timestep=timestep,
                class_labels=class_labels,
            )
            .sample
        )

    def freeze(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False


class DiffusersUNet2DCondWrapper(UNet2DConditionModel):
    """
    Wrapper for the conditional UNet2DConditionModel from diffusers.

    Bridge-aware MAAM is inserted by forward pre-hooks on selected up_blocks.
    This keeps the original SD1.5 UNet structure and weights compatible.
    """

    def __init__(self, *args, **kwargs):
        UNet2DConditionModel.__init__(self, *args, **kwargs)

        # Bridge-aware MAAM skip refinement.
        # Keep empty during SD1.5 strict loading.
        # Call enable_bridge_maam(...) after denoiser.load_state_dict(..., strict=True).
        self.use_bridge_maam = False
        self.bridge_maam_refiners = nn.ModuleDict()
        self._bridge_maam_handles = []
        self._bridge_maam_logs = []
        self.bridge_maam_use_timestep = True
        # Direction-aware Bi-LBM.
        # Keep disabled during SD1.5 strict weight loading.
        # Call enable_direction_embedding(...) after denoiser.load_state_dict(..., strict=True).
        self.use_direction_embedding = False
        self.num_directions = 0

    def forward(
            self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            conditioning: Dict[str, torch.Tensor],
            ip_adapter_cond_embedding: Optional[List[torch.Tensor]] = None,
            down_block_additional_residuals: torch.Tensor = None,
            mid_block_additional_residual: torch.Tensor = None,
            down_intrablock_additional_residuals: torch.Tensor = None,
            direction_ids: Optional[torch.Tensor] = None,
            *args,
            **kwargs,
    ):
        assert isinstance(conditioning, dict), "conditionings must be a dictionary"

        class_labels = conditioning["cond"].get("vector", None)
        crossattn = conditioning["cond"].get("crossattn", None)
        concat = conditioning["cond"].get("concat", None)

        if self.use_direction_embedding:
            if direction_ids is None:
                raise ValueError(
                    "Direction embedding is enabled, but direction_ids is None. "
                    "Pass direction_ids to the denoiser for Bi-LBM training/sampling."
                )
            if class_labels is not None:
                raise ValueError(
                    "Both conditioning['cond']['vector'] and direction_ids are provided. "
                    "This first Bi-LBM implementation reserves class_labels for direction embedding."
                )
            class_labels = direction_ids.to(device=sample.device, dtype=torch.long)

        if concat is not None:
            sample = torch.cat([sample, concat], dim=1)

        if down_intrablock_additional_residuals is not None:
            down_intrablock_additional_residuals_clone = [
                curr_residuals.clone()
                for curr_residuals in down_intrablock_additional_residuals
            ]
        else:
            down_intrablock_additional_residuals_clone = None

        if ip_adapter_cond_embedding is not None:
            added_cond_kwargs = {
                "image_embeds": [
                    ip_adapter_embedding.unsqueeze(1)
                    for ip_adapter_embedding in ip_adapter_cond_embedding
                ]
            }
        else:
            added_cond_kwargs = None

        # Reset Bridge-MAAM logs for this forward pass.
        self._bridge_maam_logs = []

        return (
            super()
            .forward(
                sample=sample,
                timestep=timestep,
                encoder_hidden_states=crossattn,
                class_labels=class_labels,
                added_cond_kwargs=added_cond_kwargs,
                down_block_additional_residuals=down_block_additional_residuals,
                mid_block_additional_residual=mid_block_additional_residual,
                down_intrablock_additional_residuals=down_intrablock_additional_residuals_clone,
            )
            .sample
        )

    # -------------------------------------------------------------------------
    # Bridge-aware MAAM utilities
    # -------------------------------------------------------------------------

    def enable_direction_embedding(
            self,
            num_directions: int = 2,
            init_scale: float = 0.0,
    ):
        """
        Enable learnable direction embeddings for bidirectional LBM.

        Direction ids:
            0: source_key -> target_key, e.g. sketch -> photo
            1: target_key -> source_key, e.g. photo -> sketch

        Implementation:
            We reuse diffusers UNet class_embedding path. When class_embedding is
            not None, diffusers adds class_emb to the timestep embedding.
            This does not modify conv_in / conv_out and keeps SD1.5 weights intact.
        """
        time_embed_dim = self._bridge_maam_time_embed_dim()
        self.class_embedding = nn.Embedding(num_directions, time_embed_dim)

        if init_scale == 0.0:
            nn.init.zeros_(self.class_embedding.weight)
        else:
            nn.init.normal_(self.class_embedding.weight, mean=0.0, std=init_scale)

        self.use_direction_embedding = True
        self.num_directions = int(num_directions)

        # Keep config consistent for checkpoints / reloads.
        if hasattr(self, "register_to_config"):
            self.register_to_config(num_class_embeds=num_directions)

    def _bridge_maam_time_embed_dim(self) -> int:
        if hasattr(self, "time_embedding") and hasattr(self.time_embedding, "linear_2"):
            return int(self.time_embedding.linear_2.out_features)

        block_out = getattr(self.config, "block_out_channels", None)
        if block_out is not None:
            return int(block_out[0] * 4)

        return 1280

    def _bridge_maam_select_up_indices(self, levels: Sequence[str]) -> List[int]:
        if isinstance(levels, str):
            levels = [levels]

        levels = list(levels)
        num_up = len(self.up_blocks)

        if "all" in levels:
            return list(range(num_up))

        selected = []

        # Diffusers up_blocks are ordered from low resolution to high resolution.
        # For SD1.5:
        #   index 0: lowest resolution
        #   index 3: highest resolution
        for level in levels:
            if level == "high":
                selected.append(num_up - 1)
            elif level == "mid":
                selected.append(max(num_up - 2, 0))
            elif level == "low":
                selected.append(max(num_up - 3, 0))
            else:
                raise ValueError(f"Unknown bridge_maam level: {level}")

        return sorted(list(set(selected)))

    def _bridge_maam_up_block_channel_info(
        self,
        up_block_idx: int,
    ) -> Tuple[int, List[int]]:
        """
        Infer:
            decoder_channels at the input of this up_block,
            skip_channels used by each ResNet inside this up_block.

        This avoids hard-coding SD1.5 channel sizes.
        """

        block_out = getattr(self.config, "block_out_channels", None)
        if block_out is None:
            raise ValueError("UNet config has no block_out_channels.")

        # Hidden states entering the first up block come from the mid block.
        current_hidden_channels = int(block_out[-1])

        for idx, block in enumerate(self.up_blocks):
            decoder_channels_for_this_block = current_hidden_channels

            if not hasattr(block, "resnets") or len(block.resnets) == 0:
                raise ValueError(f"up_block {idx} has no resnets; cannot infer channels.")

            skip_channels_for_this_block = []
            cur = current_hidden_channels

            for resnet in block.resnets:
                if not (hasattr(resnet, "conv1") and hasattr(resnet, "conv2")):
                    raise ValueError(
                        f"Cannot infer channels from resnet in up_block {idx}."
                    )

                conv1_in = int(resnet.conv1.in_channels)
                skip_c = conv1_in - int(cur)

                if skip_c <= 0:
                    # Fallback: usually the skip channels match the output channels
                    # of the current up block.
                    skip_c = int(resnet.conv2.out_channels)

                skip_channels_for_this_block.append(skip_c)

                # After this ResNet, hidden states have out_channels.
                cur = int(resnet.conv2.out_channels)

            if idx == up_block_idx:
                return decoder_channels_for_this_block, skip_channels_for_this_block

            current_hidden_channels = cur

        raise ValueError(f"Invalid up_block_idx={up_block_idx}.")

    def enable_bridge_maam(
        self,
        mode: str = "residual",
        levels: Sequence[str] = ("high",),
        attn_type: str = "scsa",
        alpha_init: float = 0.01,
        attn_bias_init: float = 2.0,
        use_timestep: bool = True,
        zero_init_timestep: bool = True,
        scsa_groups: int = 4,
        scsa_kernels: Sequence[int] = (3, 5, 7, 9),
        scsa_pool_size: int = 7,
    ):
        """
        Enable Bridge-aware MAAM skip refinement.

        Must be called after loading SD1.5 UNet weights with strict=True.
        """

        self.disable_bridge_maam()

        self.use_bridge_maam = True
        self.bridge_maam_use_timestep = bool(use_timestep)

        time_embed_dim = self._bridge_maam_time_embed_dim()
        selected_indices = self._bridge_maam_select_up_indices(levels)

        created = []

        for up_idx in selected_indices:
            decoder_channels, skip_channels_list = self._bridge_maam_up_block_channel_info(
                up_idx
            )

            unique_skip_channels = sorted(list(set(skip_channels_list)))

            for skip_channels in unique_skip_channels:
                key = f"{up_idx}_{skip_channels}"

                self.bridge_maam_refiners[key] = BridgeAwareMAAMSkipRefiner(
                    skip_channels=skip_channels,
                    decoder_channels=decoder_channels,
                    time_embed_dim=time_embed_dim,
                    mode=mode,
                    attn_type=attn_type,
                    alpha_init=alpha_init,
                    attn_bias_init=attn_bias_init,
                    zero_init_timestep=zero_init_timestep,
                    scsa_groups=scsa_groups,
                    scsa_kernels=scsa_kernels,
                    scsa_pool_size=scsa_pool_size,
                )

                created.append(
                    {
                        "up_idx": up_idx,
                        "skip_channels": skip_channels,
                        "decoder_channels": decoder_channels,
                    }
                )

            handle = self.up_blocks[up_idx].register_forward_pre_hook(
                self._make_bridge_maam_pre_hook(up_idx),
                with_kwargs=True,
            )
            self._bridge_maam_handles.append(handle)

        print(
            "[BridgeMAAM] enabled:",
            f"mode={mode}, attn_type={attn_type}, levels={list(levels)},",
            f"up_indices={selected_indices}, time_embed_dim={time_embed_dim},",
            f"created={created}",
        )

    def disable_bridge_maam(self):
        for handle in getattr(self, "_bridge_maam_handles", []):
            handle.remove()

        self._bridge_maam_handles = []
        self.bridge_maam_refiners = nn.ModuleDict()
        self.use_bridge_maam = False
        self._bridge_maam_logs = []

    def _make_bridge_maam_pre_hook(self, up_idx: int):
        def hook(module, args, kwargs):
            if not getattr(self, "use_bridge_maam", False):
                return args, kwargs

            hidden_states = kwargs.get("hidden_states", None)
            res_samples = kwargs.get("res_hidden_states_tuple", None)
            temb = kwargs.get("temb", None)

            if hidden_states is None and len(args) > 0:
                hidden_states = args[0]

            if res_samples is None or hidden_states is None:
                return args, kwargs

            if not getattr(self, "bridge_maam_use_timestep", True):
                temb = None

            new_res_samples = []

            for skip in res_samples:
                key = f"{up_idx}_{int(skip.shape[1])}"

                if key not in self.bridge_maam_refiners:
                    # Keep original skip feature if there is no matching refiner.
                    new_res_samples.append(skip)
                    continue

                refiner = self.bridge_maam_refiners[key]

                refined_skip, logs = refiner(
                    skip=skip,
                    decoder=hidden_states,
                    temb=temb,
                )

                new_res_samples.append(refined_skip)

                if not hasattr(self, "_bridge_maam_logs"):
                    self._bridge_maam_logs = []
                self._bridge_maam_logs.append(logs)

            kwargs["res_hidden_states_tuple"] = tuple(new_res_samples)

            return args, kwargs

        return hook

    def get_bridge_maam_log_dict(self, device=None):
        logs = getattr(self, "_bridge_maam_logs", [])

        if len(logs) == 0:
            if device is None:
                device = next(self.parameters()).device

            zero = torch.zeros((), device=device)

            return {
                "bridge_maam_alpha_mean": zero,
                "bridge_maam_attn_mean": zero,
                "bridge_maam_attn_std": zero,
                "bridge_maam_delta_ratio": zero,
                "bridge_maam_direct_diff_ratio": zero,
            }

        def mean_key(name: str):
            vals = [log[name].float() for log in logs]
            out = torch.stack(vals).mean()
            if device is not None:
                out = out.to(device=device)
            return out

        return {
            "bridge_maam_alpha_mean": mean_key("alpha"),
            "bridge_maam_attn_mean": mean_key("attn_mean"),
            "bridge_maam_attn_std": mean_key("attn_std"),
            "bridge_maam_delta_ratio": mean_key("delta_ratio"),
            "bridge_maam_direct_diff_ratio": mean_key("direct_diff_ratio"),
        }

    def freeze(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False