# src/lbm/models/bridge_maam.py

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _largest_divisor_leq(value: int, max_divisor: int) -> int:
    max_divisor = min(int(max_divisor), int(value))
    for d in range(max_divisor, 0, -1):
        if value % d == 0:
            return d
    return 1


class Conv1x1GnSilu(nn.Module):
    """
    Conv1x1 + GroupNorm + SiLU.

    This replaces the original MAAM Conv1x1 + BN + ReLU,
    which is more suitable for small-batch SD/LBM training.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 32,
    ):
        super().__init__()

        gn_groups = _largest_divisor_leq(out_channels, groups)

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(gn_groups, out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Conv1x1Gn(nn.Module):
    """
    Conv1x1 + GroupNorm.

    Sigmoid is intentionally not included here, because we need:
        logits + timestep_bias + attention_bias -> Sigmoid
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 32,
    ):
        super().__init__()

        gn_groups = _largest_divisor_leq(out_channels, groups)

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(gn_groups, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderAttentionGenerator(nn.Module):
    """
    MASC-Net MAAM-like decoder attention branch, adapted for LBM.

    Original MAAM:
        Decoder feature
        -> Conv1x1 + BN + ReLU
        -> Conv1x1 + BN + Sigmoid

    Ours:
        Decoder feature
        -> Conv1x1 + GN + SiLU
        -> Conv1x1 + GN
        -> + timestep bias
        -> + attention bias
        -> Sigmoid
    """

    def __init__(
        self,
        decoder_channels: int,
        skip_channels: int,
        norm_groups: int = 32,
    ):
        super().__init__()

        self.block1 = Conv1x1GnSilu(
            in_channels=decoder_channels,
            out_channels=skip_channels,
            groups=norm_groups,
        )
        self.block2 = Conv1x1Gn(
            in_channels=skip_channels,
            out_channels=skip_channels,
            groups=norm_groups,
        )

    def forward(self, decoder: torch.Tensor) -> torch.Tensor:
        return self.block2(self.block1(decoder))


class CBAMLite(nn.Module):
    """
    CBAM-style attention for ablation:
        bridge_maam_attn_type: cbam

    This keeps the original MAAM-like CA -> SA idea as an alternative
    to SCSA.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        spatial_kernel: int = 7,
    ):
        super().__init__()

        hidden = max(channels // reduction, 4)

        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )

        self.spatial = nn.Conv2d(
            2,
            1,
            kernel_size=spatial_kernel,
            padding=spatial_kernel // 2,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = F.adaptive_avg_pool2d(x, 1)
        mx = F.adaptive_max_pool2d(x, 1)

        ca = torch.sigmoid(self.channel_mlp(avg) + self.channel_mlp(mx))
        x = x * ca

        avg_sp = x.mean(dim=1, keepdim=True)
        max_sp = x.amax(dim=1, keepdim=True)

        sa = torch.sigmoid(self.spatial(torch.cat([avg_sp, max_sp], dim=1)))
        x = x * sa

        return x



class SCSAOfficialLite(nn.Module):
    """
    Dependency-free SCSA module adapted from the official implementation.

    Official SCSA dependencies include einops.rearrange and mmengine.model.BaseModule.
    This version keeps the official computation graph but removes those external
    dependencies by using pure PyTorch reshape / permute operations.

    Main official logic preserved:
        1) SMSA spatial-prior calculation:
           - x_h = mean over width, x_w = mean over height
           - split channels into 4 semantic groups
           - depth-wise Conv1d kernels [3, 5, 7, 9]
           - GroupNorm(4, dim) + gate
           - x = x * x_h_attn * x_w_attn

        2) PCSA channel attention:
           - spatial compression by AvgPool2d(window_size, stride=window_size)
             or AdaptiveAvgPool2d(1,1) when window_size == -1
           - GroupNorm(1, dim)
           - depth-wise 1x1 q/k/v Conv2d
           - reshape to [B, head_num, head_dim, H*W]
           - channel-wise self-attention
           - spatial mean -> channel gate
           - return gate * x
    """

    def __init__(
        self,
        dim: int,
        head_num: int = 8,
        window_size: int = 7,
        group_kernel_sizes: Sequence[int] = (3, 5, 7, 9),
        qkv_bias: bool = False,
        down_sample_mode: str = "avg_pool",
        attn_drop_ratio: float = 0.0,
        gate_layer: str = "sigmoid",
    ):
        super().__init__()

        if dim % 4 != 0:
            raise ValueError(
                f"Official SCSA requires dim divisible by 4, got dim={dim}."
            )

        if dim % head_num != 0:
            raise ValueError(
                f"Official SCSA requires dim divisible by head_num, "
                f"got dim={dim}, head_num={head_num}."
            )

        if len(group_kernel_sizes) != 4:
            raise ValueError(
                "Official SCSA uses four group kernel sizes, "
                f"got {group_kernel_sizes}."
            )

        if gate_layer not in {"sigmoid", "softmax"}:
            raise ValueError(f"Unsupported gate_layer={gate_layer}")

        if down_sample_mode not in {"avg_pool", "max_pool", "recombination"}:
            raise ValueError(f"Unsupported down_sample_mode={down_sample_mode}")

        self.dim = int(dim)
        self.head_num = int(head_num)
        self.head_dim = self.dim // self.head_num
        self.scaler = self.head_dim ** -0.5
        self.group_kernel_sizes = list(group_kernel_sizes)
        self.window_size = int(window_size)
        self.qkv_bias = bool(qkv_bias)
        self.down_sample_mode = down_sample_mode

        self.group_chans = self.dim // 4

        # SMSA: four semantic channel groups with local/global depth-wise 1D convs.
        self.local_dwc = nn.Conv1d(
            self.group_chans,
            self.group_chans,
            kernel_size=self.group_kernel_sizes[0],
            padding=self.group_kernel_sizes[0] // 2,
            groups=self.group_chans,
        )
        self.global_dwc_s = nn.Conv1d(
            self.group_chans,
            self.group_chans,
            kernel_size=self.group_kernel_sizes[1],
            padding=self.group_kernel_sizes[1] // 2,
            groups=self.group_chans,
        )
        self.global_dwc_m = nn.Conv1d(
            self.group_chans,
            self.group_chans,
            kernel_size=self.group_kernel_sizes[2],
            padding=self.group_kernel_sizes[2] // 2,
            groups=self.group_chans,
        )
        self.global_dwc_l = nn.Conv1d(
            self.group_chans,
            self.group_chans,
            kernel_size=self.group_kernel_sizes[3],
            padding=self.group_kernel_sizes[3] // 2,
            groups=self.group_chans,
        )

        self.sa_gate = nn.Softmax(dim=2) if gate_layer == "softmax" else nn.Sigmoid()
        self.norm_h = nn.GroupNorm(4, self.dim)
        self.norm_w = nn.GroupNorm(4, self.dim)

        # PCSA.
        self.conv_d = nn.Identity()
        self.norm = nn.GroupNorm(1, self.dim)

        self.q = nn.Conv2d(
            in_channels=self.dim,
            out_channels=self.dim,
            kernel_size=1,
            bias=self.qkv_bias,
            groups=self.dim,
        )
        self.k = nn.Conv2d(
            in_channels=self.dim,
            out_channels=self.dim,
            kernel_size=1,
            bias=self.qkv_bias,
            groups=self.dim,
        )
        self.v = nn.Conv2d(
            in_channels=self.dim,
            out_channels=self.dim,
            kernel_size=1,
            bias=self.qkv_bias,
            groups=self.dim,
        )

        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.ca_gate = nn.Softmax(dim=1) if gate_layer == "softmax" else nn.Sigmoid()

        if self.window_size == -1:
            self.down_func = nn.AdaptiveAvgPool2d((1, 1))
        else:
            if down_sample_mode == "recombination":
                self.down_func = self.space_to_chans
                self.conv_d = nn.Conv2d(
                    in_channels=self.dim * self.window_size ** 2,
                    out_channels=self.dim,
                    kernel_size=1,
                    bias=False,
                )
            elif down_sample_mode == "avg_pool":
                self.down_func = nn.AvgPool2d(
                    kernel_size=(self.window_size, self.window_size),
                    stride=self.window_size,
                )
            elif down_sample_mode == "max_pool":
                self.down_func = nn.MaxPool2d(
                    kernel_size=(self.window_size, self.window_size),
                    stride=self.window_size,
                )

    def space_to_chans(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pure-PyTorch replacement for the official recombination branch.

        Input:
            x: [B, C, H, W]
        Output:
            y: [B, C * window_size^2, ceil(H/ws), ceil(W/ws)]
        """
        b, c, h, w = x.shape
        ws = self.window_size

        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        b, c, h_pad, w_pad = x.shape
        h_new = h_pad // ws
        w_new = w_pad // ws

        x = x.view(b, c, h_new, ws, w_new, ws)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(b, c * ws * ws, h_new, w_new)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Official SCSA forward logic.

        Input:
            x: [B, C, H, W]
        Output:
            out: [B, C, H, W]
        """
        b, c, h_, w_ = x.size()

        # -------------------------
        # SMSA: spatial attention priority calculation.
        # -------------------------
        # Official: x_h = x.mean(dim=3), shape [B, C, H]
        x_h = x.mean(dim=3)
        l_x_h, g_x_h_s, g_x_h_m, g_x_h_l = torch.split(
            x_h,
            self.group_chans,
            dim=1,
        )

        # Official: x_w = x.mean(dim=2), shape [B, C, W]
        x_w = x.mean(dim=2)
        l_x_w, g_x_w_s, g_x_w_m, g_x_w_l = torch.split(
            x_w,
            self.group_chans,
            dim=1,
        )

        x_h_attn = self.sa_gate(
            self.norm_h(
                torch.cat(
                    (
                        self.local_dwc(l_x_h),
                        self.global_dwc_s(g_x_h_s),
                        self.global_dwc_m(g_x_h_m),
                        self.global_dwc_l(g_x_h_l),
                    ),
                    dim=1,
                )
            )
        )
        x_h_attn = x_h_attn.view(b, c, h_, 1)

        x_w_attn = self.sa_gate(
            self.norm_w(
                torch.cat(
                    (
                        self.local_dwc(l_x_w),
                        self.global_dwc_s(g_x_w_s),
                        self.global_dwc_m(g_x_w_m),
                        self.global_dwc_l(g_x_w_l),
                    ),
                    dim=1,
                )
            )
        )
        x_w_attn = x_w_attn.view(b, c, 1, w_)

        x = x * x_h_attn * x_w_attn

        # -------------------------
        # PCSA: channel attention based on self-attention.
        # -------------------------
        # Official SCSA uses AvgPool2d(window_size, stride=window_size).
        # When B-MAAM is enabled on all SD1.5 up_blocks, the lowest-resolution
        # skip feature can be 4x4. AvgPool2d(7, 7) would produce a 0x0 output.
        #
        # For such small feature maps, fall back to global average pooling.
        # This keeps the official PCSA path unchanged after downsampling:
        #   downsample -> conv_d -> GN -> depth-wise q/k/v -> channel attention.
        if (
                self.window_size != -1
                and self.down_sample_mode in {"avg_pool", "max_pool"}
                and (h_ < self.window_size or w_ < self.window_size)
        ):
            y = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        else:
            y = self.down_func(x)

        y = self.conv_d(y)

        _, _, h_, w_ = y.size()

        y = self.norm(y)

        q = self.q(y)
        k = self.k(y)
        v = self.v(y)

        n = h_ * w_

        # Official einops:
        # q = rearrange(q, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', ...)
        q = q.view(b, self.head_num, self.head_dim, n)
        k = k.view(b, self.head_num, self.head_dim, n)
        v = v.view(b, self.head_num, self.head_dim, n)

        # [B, head_num, head_dim, head_dim]
        attn = q @ k.transpose(-2, -1) * self.scaler
        attn = self.attn_drop(attn.softmax(dim=-1))

        # [B, head_num, head_dim, N]
        attn = attn @ v

        # Official inverse rearrange:
        # [B, head_num, head_dim, H*W] -> [B, C, H, W]
        attn = attn.reshape(b, self.dim, h_, w_)

        # [B, C, 1, 1]
        attn = attn.mean((2, 3), keepdim=True)
        attn = self.ca_gate(attn)

        return attn * x


class SCSAAttention(nn.Module):
    """
    Official-lite SCSA wrapper.

    This is a dependency-free port of the official SCSA implementation:
        - removes mmengine.BaseModule
        - removes einops.rearrange
        - keeps the official SMSA -> PCSA order and computation.

    Existing B-MAAM code calls this class through:
        SCSAAttention(channels=..., groups=..., kernels=..., pool_size=...)
    For compatibility, `groups` is accepted but official SCSA uses four semantic
    groups internally.
    """

    def __init__(
        self,
        channels: int,
        groups: int = 4,
        kernels: Sequence[int] = (3, 5, 7, 9),
        pool_size: int = 7,
        head_num: int = 8,
        qkv_bias: bool = False,
        down_sample_mode: str = "avg_pool",
        attn_drop_ratio: float = 0.0,
        gate_layer: str = "sigmoid",
    ):
        super().__init__()

        if groups != 4:
            raise ValueError(
                f"Official SCSA uses 4 semantic channel groups, got groups={groups}."
            )

        # SD1.5 skip channels such as 320/640/1280 are divisible by 8.
        # For other channel sizes, reduce head_num to the largest valid divisor.
        if channels % head_num != 0:
            head_num = _largest_divisor_leq(channels, head_num)

        self.scsa = SCSAOfficialLite(
            dim=channels,
            head_num=head_num,
            window_size=pool_size,
            group_kernel_sizes=kernels,
            qkv_bias=qkv_bias,
            down_sample_mode=down_sample_mode,
            attn_drop_ratio=attn_drop_ratio,
            gate_layer=gate_layer,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scsa(x)


class BridgeAwareMAAMSkipRefiner(nn.Module):
    """
    Bridge-aware SCSA-MAAM skip refinement.

    Input:
        skip:
            E_i, encoder skip feature, [B, C, H, W]

        decoder:
            D_i, current decoder feature, [B, C_d, H_d, W_d]

        temb:
            LBM / SD UNet time embedding, [B, T]

    Attention map:
        A_i = sigmoid(DecoderConv(D_i) + MLP(t) + attention_bias)

    MAAM-selected skip:
        X_i = A_i * E_i

    SCSA-enhanced skip:
        F_i = SCSA(X_i)

    Output:
        residual:
            E_i' = E_i + alpha * (F_i - E_i)

        direct:
            E_i' = F_i
    """

    def __init__(
        self,
        skip_channels: int,
        decoder_channels: int,
        time_embed_dim: int,
        mode: str = "residual",
        attn_type: str = "scsa",
        alpha_init: float = 0.01,
        attn_bias_init: float = 2.0,
        zero_init_timestep: bool = True,
        scsa_groups: int = 4,
        scsa_kernels: Sequence[int] = (3, 5, 7, 9),
        scsa_pool_size: int = 7,
        norm_groups: int = 32,
    ):
        super().__init__()

        if mode not in {"residual", "direct"}:
            raise ValueError(f"Unsupported bridge_maam mode: {mode}")

        if attn_type not in {"scsa", "cbam", "none"}:
            raise ValueError(f"Unsupported bridge_maam attn_type: {attn_type}")

        self.skip_channels = int(skip_channels)
        self.decoder_channels = int(decoder_channels)
        self.time_embed_dim = int(time_embed_dim)
        self.mode = mode
        self.attn_type = attn_type

        self.decoder_to_logits = DecoderAttentionGenerator(
            decoder_channels=self.decoder_channels,
            skip_channels=self.skip_channels,
            norm_groups=norm_groups,
        )

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.skip_channels, bias=True),
        )

        if zero_init_timestep:
            nn.init.zeros_(self.time_mlp[-1].weight)
            nn.init.zeros_(self.time_mlp[-1].bias)

        self.attn_bias = nn.Parameter(
            torch.full((1, self.skip_channels, 1, 1), float(attn_bias_init))
        )

        self.alpha = nn.Parameter(torch.full((1,), float(alpha_init)))

        if attn_type == "scsa":
            self.post_attn = SCSAAttention(
                channels=self.skip_channels,
                groups=scsa_groups,
                kernels=scsa_kernels,
                pool_size=scsa_pool_size,
            )
        elif attn_type == "cbam":
            self.post_attn = CBAMLite(channels=self.skip_channels)
        else:
            self.post_attn = nn.Identity()

    def forward(
        self,
        skip: torch.Tensor,
        decoder: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        e = skip
        d = decoder

        if d.shape[-2:] != e.shape[-2:]:
            d = F.interpolate(
                d,
                size=e.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        logits = self.decoder_to_logits(d)

        if temb is not None:
            if temb.ndim > 2:
                temb = temb.view(temb.shape[0], -1)

            time_bias = self.time_mlp(temb.to(dtype=logits.dtype))
            time_bias = time_bias.view(e.shape[0], self.skip_channels, 1, 1)
        else:
            time_bias = torch.zeros(
                e.shape[0],
                self.skip_channels,
                1,
                1,
                device=e.device,
                dtype=logits.dtype,
            )

        attn = torch.sigmoid(
            logits
            + time_bias.to(dtype=logits.dtype)
            + self.attn_bias.to(device=logits.device, dtype=logits.dtype)
        )

        selected = attn.to(dtype=e.dtype) * e
        f = self.post_attn(selected)

        if self.mode == "residual":
            alpha = self.alpha.to(device=e.device, dtype=e.dtype).view(1, 1, 1, 1)
            refined = e + alpha * (f - e)
        else:
            refined = f

        delta = refined - e
        direct_diff = f - e

        denom = e.detach().float().abs().mean().clamp_min(1e-6)
        delta_ratio = delta.detach().float().abs().mean() / denom
        direct_diff_ratio = direct_diff.detach().float().abs().mean() / denom

        logs = {
            "alpha": self.alpha.detach().float().mean(),
            "attn_mean": attn.detach().float().mean(),
            "attn_std": attn.detach().float().std(unbiased=False),
            "delta_ratio": delta_ratio.detach().float(),
            "direct_diff_ratio": direct_diff_ratio.detach().float(),
        }

        return refined, logs