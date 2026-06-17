import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from lbm.models.id_encoder.model_irse import Backbone


class ArcFaceIDLoss(nn.Module):
    """
    ArcFace/IR-SE50 identity loss.

    Input:
        y_hat: generated image, range [-1, 1], shape [B, 3, H, W]
        y:     target image,    range [-1, 1], shape [B, 3, H, W]

    Loss:
        1 - cosine_similarity(ArcFace(y_hat), ArcFace(y))
    """

    def __init__(
        self,
        model_path: str,
        crop_face: bool = True,
    ):
        super().__init__()

        if model_path is None or not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"ArcFace model not found: {model_path}. "
                f"Please put model_ir_se50.pth at this path."
            )

        print(f"[ArcFaceIDLoss] Loading ArcFace model from: {model_path}")

        self.facenet = Backbone(
            input_size=112,
            num_layers=50,
            drop_ratio=0.6,
            mode="ir_se",
        )

        state_dict = torch.load(model_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # remove possible 'module.' prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[len("module."):]
            new_state_dict[k] = v

        self.facenet.load_state_dict(new_state_dict, strict=False)
        self.facenet.float()
        self.facenet.eval()

        for p in self.facenet.parameters():
            p.requires_grad = False

        self.face_pool = nn.AdaptiveAvgPool2d((112, 112))
        self.crop_face = crop_face

    def extract_feats(self, x: torch.Tensor) -> torch.Tensor:
        # ArcFace/IR-SE50 用 float32 计算，避免 bf16 mixed precision 下 BatchNorm/Conv dtype 冲突
        device_type = x.device.type

        self.facenet.float()
        self.facenet.eval()

        with torch.autocast(device_type=device_type, enabled=False):
            x = x.float().clamp(-1, 1)

            # DECP uses this crop for 256x256 aligned face images:
            # x[:, :, 35:223, 32:220]
            if self.crop_face and x.shape[-2] >= 224 and x.shape[-1] >= 224:
                x = x[:, :, 35:223, 32:220]

            x = self.face_pool(x)
            feats = self.facenet(x)
            feats = F.normalize(feats, p=2, dim=1)

        return feats

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_hat_feats = self.extract_feats(y_hat)
        with torch.no_grad():
            y_feats = self.extract_feats(y)

        sim = torch.sum(y_hat_feats * y_feats, dim=1)
        loss = 1.0 - sim
        return loss.mean()