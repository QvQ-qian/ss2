from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

'''
这版和 T2I-Adapter 的差别是：
T2I-Adapter 默认通道通常是 [320,640,1280,1280]，每个 block 默认 2 个 ResNet-like block；
这里为了小数据集和人脸 parsing 做了轻量化，用 [64,128,256,512] 内部通道，再用 projection 对齐 UNet 通道。
T2I-Adapter 官方实现确实是多尺度输出，并默认 num_res_blocks=2、channels=[320,640,1280,1280]。
'''

def zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class GNActConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, groups=8):
        super().__init__()
        groups = min(groups, out_ch)
        while out_ch % groups != 0 and groups > 1:
            groups -= 1

        self.block = nn.Sequential(
            nn.GroupNorm(groups, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding),
        )

    def forward(self, x):
        return self.block(x)


class FaceAdapterResBlock(nn.Module):
    """
    Lightweight face-specific residual block:
    GN + SiLU + 3x3 Conv + GN + SiLU + 3x3 Conv.
    """

    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        groups = min(groups, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1

        self.norm1 = nn.GroupNorm(groups, channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(groups, channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.act2(self.norm2(h)))
        return x + h


class ParseStem(nn.Module):
    """
    Parse one-hot [B,19,256,256] -> [B,64,32,32].
    """

    def __init__(self, in_channels=19, stem_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),  # 128
            nn.GroupNorm(8, 32),
            nn.SiLU(),

            nn.Conv2d(32, stem_channels, kernel_size=3, stride=2, padding=1),  # 64
            nn.GroupNorm(8, stem_channels),
            nn.SiLU(),

            nn.Conv2d(stem_channels, stem_channels, kernel_size=3, stride=2, padding=1),  # 32
            nn.GroupNorm(8, stem_channels),
            nn.SiLU(),
        )

    def forward(self, parse):
        return self.net(parse)


class ImageConditionStem(nn.Module):
    """
    Image-like condition [B,C,256,256] -> [B,64,32,32].

    Used for:
      - sketch condition
      - coarse face condition
      - future image-like priors
    """

    def __init__(self, in_channels=3, stem_channels=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),  # 128
            nn.GroupNorm(8, 32),
            nn.SiLU(),

            nn.Conv2d(32, stem_channels, kernel_size=3, stride=2, padding=1),  # 64
            nn.GroupNorm(8, stem_channels),
            nn.SiLU(),

            nn.Conv2d(stem_channels, stem_channels, kernel_size=3, stride=2, padding=1),  # 32
            nn.GroupNorm(8, stem_channels),
            nn.SiLU(),
        )

    def forward(self, image_condition):
        return self.net(image_condition)


# Backward-compatible alias.
# If some old code imports CoarseFaceStem, it will still work.
CoarseFaceStem = ImageConditionStem


class FaceConditionalAdapter(nn.Module):
    """
    Lightweight face parsing adapter inspired by T2I-Adapter.

    V1:
        parse -> parse stem -> shared trunk -> multi-scale residuals

    Future:
        parse stem + coarse face stem -> fusion -> shared trunk
    """

    def __init__(
            self,
            parse_in_channels: int = 19,
            sketch_in_channels: int = 3,
            coarse_in_channels: int = 3,
            use_parse: bool = True,
            use_sketch: bool = False,
            use_coarse: bool = False,
            internal_channels: Tuple[int, int, int, int] = (64, 128, 256, 512),
            unet_channels: Tuple[int, int, int, int] = (320, 640, 1280, 1280),
            include_mid: bool = True,
            zero_init: bool = True,
            condition_dropout: float = 0.0,
            use_scale_gates: bool = True,
            gate_init: float = 1.0,
    ):
        super().__init__()

        self.use_parse = use_parse
        self.use_sketch = use_sketch
        self.use_coarse = use_coarse
        self.include_mid = include_mid
        self.condition_dropout = float(condition_dropout)

        c0, c1, c2, c3 = internal_channels

        if self.use_parse:
            self.parse_stem = ParseStem(parse_in_channels, c0)
        else:
            self.parse_stem = None

        if self.use_sketch:
            self.sketch_stem = ImageConditionStem(sketch_in_channels, c0)
        else:
            self.sketch_stem = None

        if self.use_coarse:
            self.coarse_stem = ImageConditionStem(coarse_in_channels, c0)
        else:
            self.coarse_stem = None

        n_branches = int(self.use_parse) + int(self.use_sketch) + int(self.use_coarse)
        if n_branches <= 0:
            raise ValueError(
                "FaceConditionalAdapter requires at least one condition branch. "
                "Please enable use_parse, use_sketch, or use_coarse."
            )

        if n_branches > 1:
            self.fuse = nn.Sequential(
                nn.Conv2d(c0 * n_branches, c0, kernel_size=1),
                nn.GroupNorm(8, c0),
                nn.SiLU(),
            )
        else:
            self.fuse = None

        # 32x32
        self.stage0 = nn.Sequential(
            FaceAdapterResBlock(c0),
            FaceAdapterResBlock(c0),
        )

        # 16x16
        self.down1 = nn.Conv2d(c0, c1, kernel_size=3, stride=2, padding=1)
        self.stage1 = nn.Sequential(
            FaceAdapterResBlock(c1),
            FaceAdapterResBlock(c1),
        )

        # 8x8
        self.down2 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1)
        self.stage2 = nn.Sequential(
            FaceAdapterResBlock(c2),
            FaceAdapterResBlock(c2),
        )

        # 4x4
        self.down3 = nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1)
        self.stage3 = nn.Sequential(
            FaceAdapterResBlock(c3),
            FaceAdapterResBlock(c3),
        )

        self.mid = FaceAdapterResBlock(c3) if include_mid else None

        self.proj0 = nn.Conv2d(c0, unet_channels[0], kernel_size=1)
        self.proj1 = nn.Conv2d(c1, unet_channels[1], kernel_size=1)
        self.proj2 = nn.Conv2d(c2, unet_channels[2], kernel_size=1)
        self.proj3 = nn.Conv2d(c3, unet_channels[3], kernel_size=1)
        self.proj_mid = nn.Conv2d(c3, unet_channels[3], kernel_size=1) if include_mid else None

        # Per-scale learnable gates for adapter residual strength.
        # gate_init=1.0 is safer than 0.0 because proj layers are already zero-init.
        self.use_scale_gates = bool(use_scale_gates)
        if self.use_scale_gates:
            self.down_gates = nn.Parameter(
                torch.full((4,), float(gate_init), dtype=torch.float32)
            )
            self.mid_gate = (
                nn.Parameter(torch.tensor(float(gate_init), dtype=torch.float32))
                if include_mid
                else None
            )
        else:
            self.register_buffer("down_gates", torch.ones(4, dtype=torch.float32))
            self.mid_gate = None

        if zero_init:
            zero_module(self.proj0)
            zero_module(self.proj1)
            zero_module(self.proj2)
            zero_module(self.proj3)
            if self.proj_mid is not None:
                zero_module(self.proj_mid)

    def _maybe_drop_condition(self, x):
        if not self.training or self.condition_dropout <= 0:
            return x

        if torch.rand((), device=x.device) < self.condition_dropout:
            return torch.zeros_like(x)

        return x

    def forward(
            self,
            parse: Optional[torch.Tensor] = None,
            sketch: Optional[torch.Tensor] = None,
            coarse_face: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:

        features = []

        if self.use_parse:
            if parse is None:
                raise ValueError(
                    "FaceConditionalAdapter requires parse input when use_parse=True."
                )
            parse = self._maybe_drop_condition(parse)
            features.append(self.parse_stem(parse))

        if self.use_sketch:
            if sketch is None:
                raise ValueError(
                    "FaceConditionalAdapter requires sketch input when use_sketch=True."
                )
            sketch = self._maybe_drop_condition(sketch)
            features.append(self.sketch_stem(sketch))

        if self.use_coarse:
            if coarse_face is None:
                raise ValueError(
                    "FaceConditionalAdapter requires coarse_face input when use_coarse=True."
                )
            coarse_face = self._maybe_drop_condition(coarse_face)
            features.append(self.coarse_stem(coarse_face))

        if len(features) == 0:
            raise ValueError("No condition features were provided to FaceConditionalAdapter.")

        if len(features) == 1:
            x = features[0]
        else:
            x = self.fuse(torch.cat(features, dim=1))

        t0 = self.stage0(x)  # [B,64,32,32]
        t1 = self.stage1(self.down1(t0))  # [B,128,16,16]
        t2 = self.stage2(self.down2(t1))  # [B,256,8,8]
        t3 = self.stage3(self.down3(t2))  # [B,512,4,4]

        down = [
            self.proj0(t0),
            self.proj1(t1),
            self.proj2(t2),
            self.proj3(t3),
        ]

        # Apply per-scale gates.
        if self.use_scale_gates:
            gates = self.down_gates.to(device=down[0].device, dtype=down[0].dtype)
            down = [
                down[0] * gates[0],
                down[1] * gates[1],
                down[2] * gates[2],
                down[3] * gates[3],
            ]

        mid = None
        if self.include_mid:
            tm = self.mid(t3)
            mid = self.proj_mid(tm)

            if self.use_scale_gates and self.mid_gate is not None:
                mid = mid * self.mid_gate.to(device=mid.device, dtype=mid.dtype)

        return {
            "down": down,
            "mid": mid,
            "down_gates": self.down_gates.detach().float(),
            "mid_gate": None if self.mid_gate is None else self.mid_gate.detach().float(),
        }