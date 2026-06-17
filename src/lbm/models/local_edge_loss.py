# src/lbm/models/local_edge_loss.py

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_FACE_PARSE_PARTS: Dict[str, List[int]] = {
    # 19-class BiSeNet/CelebAMask-HQ style labels:
    # 2 left_brow, 3 right_brow, 4 left_eye, 5 right_eye
    # 注意：这里故意不包含 6 eye_glass
    "eye": [2, 3, 4, 5],

    # 如果你后面想只约束真正眼睛，不约束眉毛，可以用 eye_core
    "eye_core": [4, 5],

    # 10 nose
    "nose": [10],

    # 11 mouth, 12 upper_lip, 13 lower_lip
    "mouth": [11, 12, 13],
}


class SobelMagnitude(nn.Module):
    """
    Fixed Sobel edge magnitude.

    Input:
        x: [B,3,H,W] or [B,1,H,W], usually in [0,1]

    Output:
        edge: [B,1,H,W]
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)

        sobel_x = torch.tensor(
            [
                [-1.0, 0.0, 1.0],
                [-2.0, 0.0, 2.0],
                [-1.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [
                [-1.0, -2.0, -1.0],
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 1.0],
            ],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()

        if x.ndim != 4:
            raise ValueError(f"SobelMagnitude expects [B,C,H,W], got {tuple(x.shape)}")

        if x.shape[1] == 3:
            r = x[:, 0:1]
            g = x[:, 1:2]
            b = x[:, 2:3]
            x = 0.2989 * r + 0.5870 * g + 0.1140 * b
        elif x.shape[1] != 1:
            raise ValueError(f"Unsupported channel count for SobelMagnitude: {x.shape[1]}")

        sobel_x = self.sobel_x.to(device=x.device, dtype=torch.float32)
        sobel_y = self.sobel_y.to(device=x.device, dtype=torch.float32)

        gx = F.conv2d(x, sobel_x, padding=1)
        gy = F.conv2d(x, sobel_y, padding=1)
        edge = torch.sqrt(gx.square() + gy.square() + self.eps)

        return edge.float()


class LocalEdgeLoss(nn.Module):
    """
    Local face-part edge consistency loss guided by face parsing masks.

    Recommended first experiment:
        parts=["eye"]
        exclude_labels=[6]   # exclude eye_glass

    Loss:
        mean( |Sobel(pred) - Sobel(target)| * local_part_mask )

    pred / target:
        [B,3,H,W], expected in [-1,1]

    parse:
        [B,19,H,W] one-hot
        or [B,1,H,W] label map
        or [B,H,W] label map
    """

    def __init__(
        self,
        parts: Sequence[str] = ("eye",),
        num_classes: int = 19,
        part_indices: Optional[Dict[str, List[int]]] = None,
        dilate_kernel: int = 7,
        exclude_labels: Optional[Sequence[int]] = None,
        exclude_dilate_kernel: int = 7,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.parts = list(parts)
        self.num_classes = int(num_classes)
        self.part_indices = part_indices or DEFAULT_FACE_PARSE_PARTS
        self.dilate_kernel = int(dilate_kernel)
        self.exclude_labels = None if exclude_labels is None else [int(x) for x in exclude_labels]
        self.exclude_dilate_kernel = int(exclude_dilate_kernel)
        self.eps = float(eps)

        for part in self.parts:
            if part not in self.part_indices:
                raise ValueError(
                    f"Unknown local edge part: {part}. "
                    f"Available parts: {list(self.part_indices.keys())}"
                )

        self.sobel = SobelMagnitude(eps=eps)

        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    @staticmethod
    def _to_01(x: torch.Tensor) -> torch.Tensor:
        return (x.float().clamp(-1.0, 1.0) + 1.0) * 0.5

    @staticmethod
    def _dilate(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
        if kernel_size is None or int(kernel_size) <= 1:
            return mask

        k = int(kernel_size)
        if k % 2 == 0:
            k += 1

        return F.max_pool2d(mask, kernel_size=k, stride=1, padding=k // 2)

    def _parse_to_label(self, parse: torch.Tensor) -> torch.Tensor:
        """
        Convert parse to label map [B,H,W].
        """
        if parse.ndim == 4 and parse.shape[1] == self.num_classes:
            label = parse.argmax(dim=1)

        elif parse.ndim == 4 and parse.shape[1] == 1:
            label = parse[:, 0].long()

        elif parse.ndim == 3:
            label = parse.long()

        else:
            raise ValueError(
                f"Unsupported parse shape for LocalEdgeLoss: {tuple(parse.shape)}"
            )

        return label.long()

    def _labels_to_mask(
        self,
        label: torch.Tensor,
        labels: Sequence[int],
    ) -> torch.Tensor:
        """
        label: [B,H,W]
        return: [B,1,H,W]
        """
        mask = torch.zeros_like(label, dtype=torch.float32, device=label.device)

        for idx in labels:
            mask = torch.maximum(mask, (label == int(idx)).float())

        return mask.unsqueeze(1)

    def _build_part_mask(
        self,
        parse: torch.Tensor,
        target_hw,
        device,
    ) -> torch.Tensor:
        """
        Build local mask [B,1,H,W].
        """
        label = self._parse_to_label(parse.to(device=device))

        selected_indices: List[int] = []
        for part in self.parts:
            selected_indices.extend(self.part_indices[part])

        selected_indices = sorted(list(set(selected_indices)))

        include_mask = self._labels_to_mask(label, selected_indices)

        if include_mask.shape[-2:] != target_hw:
            include_mask = F.interpolate(
                include_mask,
                size=target_hw,
                mode="nearest",
            )

        include_mask = self._dilate(include_mask, self.dilate_kernel)

        # Optional: exclude glasses region after dilation.
        # This is important because glasses have very strong edges
        # and can dominate eye-region edge loss.
        if self.exclude_labels is not None and len(self.exclude_labels) > 0:
            exclude_mask = self._labels_to_mask(label, self.exclude_labels)

            if exclude_mask.shape[-2:] != target_hw:
                exclude_mask = F.interpolate(
                    exclude_mask,
                    size=target_hw,
                    mode="nearest",
                )

            exclude_mask = self._dilate(exclude_mask, self.exclude_dilate_kernel)
            include_mask = include_mask * (1.0 - exclude_mask.clamp(0.0, 1.0))

        return include_mask.clamp(0.0, 1.0)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        parse: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ):
        device = pred.device

        pred_01 = self._to_01(pred)
        target_01 = self._to_01(target)

        pred_edge = self.sobel(pred_01)
        target_edge = self.sobel(target_01)

        local_mask = self._build_part_mask(
            parse=parse,
            target_hw=pred_edge.shape[-2:],
            device=device,
        )

        if valid_mask is not None:
            valid_mask = valid_mask.to(device=device, dtype=torch.float32)

            if valid_mask.ndim != 4:
                raise ValueError(f"valid_mask should be [B,C,H,W], got {tuple(valid_mask.shape)}")

            if valid_mask.shape[1] != 1:
                valid_mask = valid_mask[:, :1]

            if valid_mask.shape[-2:] != pred_edge.shape[-2:]:
                valid_mask = F.interpolate(
                    valid_mask,
                    size=pred_edge.shape[-2:],
                    mode="nearest",
                )

            local_mask = local_mask * valid_mask

        diff = torch.abs(pred_edge - target_edge)

        denom = local_mask.sum(dim=(1, 2, 3)).clamp_min(self.eps)
        per_sample_loss = (diff * local_mask).sum(dim=(1, 2, 3)) / denom
        loss = per_sample_loss.mean()

        if return_dict:
            return loss, {
                "local_edge_loss": loss.detach(),
                "local_edge_mask_mean": local_mask.detach().float().mean(),
            }

        return loss