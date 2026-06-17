# src/lbm/models/ea_dists_loss.py

from __future__ import annotations

from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelEdgeDetector(nn.Module):
    """
    Fixed Sobel edge detector.

    Input:
        x: [B, 3, H, W] or [B, 1, H, W], expected in [0, 1]

    Output:
        edge magnitude: [B, 1, H, W]
    """

    def __init__(self, eps: float = 1e-6, normalize: bool = True):
        super().__init__()
        self.eps = float(eps)
        self.normalize = bool(normalize)

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
            raise ValueError(
                f"SobelEdgeDetector expects [B,C,H,W], got shape={tuple(x.shape)}"
            )

        if x.shape[1] == 3:
            # RGB -> grayscale.
            r = x[:, 0:1]
            g = x[:, 1:2]
            b = x[:, 2:3]
            x = 0.2989 * r + 0.5870 * g + 0.1140 * b
        elif x.shape[1] != 1:
            raise ValueError(
                f"Unsupported channel count for edge detector: {x.shape[1]}"
            )

        # Buffers may be moved/cast by Lightning/FSDP, so force dtype/device here.
        sobel_x = self.sobel_x.to(device=x.device, dtype=torch.float32)
        sobel_y = self.sobel_y.to(device=x.device, dtype=torch.float32)

        edge_x = F.conv2d(x, sobel_x, padding=1)
        edge_y = F.conv2d(x, sobel_y, padding=1)
        edge = torch.sqrt(edge_x.square() + edge_y.square() + self.eps)

        if self.normalize:
            max_val = edge.amax(dim=(2, 3), keepdim=True).clamp_min(self.eps)
            edge = edge / max_val

        return edge.float()


class EADISTSLoss(nn.Module):
    """
    Edge-Aware DISTS loss:

        EA-DISTS(pred, target)
            = DISTS(pred, target)
            + edge_weight * DISTS(Sobel(pred), Sobel(target))

    Current LBM decoded images are expected in [-1, 1].
    Internally this loss converts them to [0, 1] and forces float32,
    because pyiqa-DISTS is not safe under bf16-mixed autocast.
    """

    def __init__(
        self,
        edge_weight: float = 1.0,
        use_edge: bool = True,
        edge_to_rgb: bool = True,
        edge_normalize: bool = True,
        resize_to: Optional[int] = None,
    ):
        super().__init__()

        try:
            import pyiqa
        except ImportError as e:
            raise ImportError(
                "EADISTSLoss requires pyiqa. Please install it with: pip install pyiqa"
            ) from e

        self.edge_weight = float(edge_weight)
        self.use_edge = bool(use_edge)
        self.edge_to_rgb = bool(edge_to_rgb)
        self.resize_to = resize_to

        self.dists = pyiqa.create_metric("dists", as_loss=True)
        self.edge_model = SobelEdgeDetector(normalize=edge_normalize)

        self._freeze_loss_networks()

    def _freeze_loss_networks(self) -> None:
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    def _force_float32(self, device: torch.device) -> None:
        """
        Force DISTS and Sobel modules to float32.

        This is necessary because the main LBM training uses bf16-mixed precision.
        Even with autocast disabled, some submodules/buffers may still be bf16
        after Lightning/FSDP moves the parent model.
        """
        self.dists.to(device=device)
        self.dists.float()

        # pyiqa.create_metric returns an InferenceModel; the real model is usually .net.
        if hasattr(self.dists, "net"):
            self.dists.net.to(device=device)
            self.dists.net.float()

        self.edge_model.to(device=device)
        self.edge_model.float()

        self._freeze_loss_networks()

    @staticmethod
    def _to_01(x: torch.Tensor) -> torch.Tensor:
        # LBM decoded_prediction / model_input are in [-1, 1].
        return (x.float().clamp(-1.0, 1.0) + 1.0) * 0.5

    def _maybe_resize(self, x: torch.Tensor) -> torch.Tensor:
        if self.resize_to is None:
            return x

        size = int(self.resize_to)
        if x.shape[-2:] == (size, size):
            return x

        return F.interpolate(
            x,
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        )

    def _prepare_for_dists(self, x: torch.Tensor) -> torch.Tensor:
        x = self._maybe_resize(x)
        return x.to(dtype=torch.float32).contiguous()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        return_dict: bool = False,
    ):
        device = pred.device

        # Critical fix for bf16-mixed training.
        self._force_float32(device)

        if device.type == "cuda":
            autocast_ctx = torch.amp.autocast("cuda", enabled=False)
        else:
            autocast_ctx = nullcontext()

        with autocast_ctx:
            pred_01 = self._prepare_for_dists(self._to_01(pred))
            target_01 = self._prepare_for_dists(self._to_01(target))

            dists_loss = self.dists(
                pred_01.to(device=device, dtype=torch.float32),
                target_01.to(device=device, dtype=torch.float32),
            ).mean()

            if self.use_edge and self.edge_weight > 0:
                pred_edge = self.edge_model(pred_01)
                target_edge = self.edge_model(target_01)

                if self.edge_to_rgb:
                    pred_edge = pred_edge.repeat(1, 3, 1, 1)
                    target_edge = target_edge.repeat(1, 3, 1, 1)

                pred_edge = self._prepare_for_dists(pred_edge)
                target_edge = self._prepare_for_dists(target_edge)

                edge_loss = self.dists(
                    pred_edge.to(device=device, dtype=torch.float32),
                    target_edge.to(device=device, dtype=torch.float32),
                ).mean()
            else:
                edge_loss = torch.zeros_like(dists_loss)

            total_loss = dists_loss + self.edge_weight * edge_loss

        if return_dict:
            return total_loss, {
                "dists": dists_loss.detach(),
                "edge": edge_loss.detach(),
                "total": total_loss.detach(),
            }

        return total_loss