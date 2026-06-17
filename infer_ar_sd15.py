import os
import argparse
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.utils import save_image

from examples.training.train_lbm_surface_sd15 import get_model


IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]


def load_image(path, size=256):
    img = Image.open(path).convert("RGB")
    tfm = transforms.Compose([
        transforms.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return tfm(img).unsqueeze(0)


def tensor_to_image(x):
    # [-1, 1] -> [0, 1]
    return (x.clamp(-1, 1) + 1) / 2


def load_lbm_model(config_path, ckpt_path, device="cuda"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    model = get_model(
        backbone_signature=cfg["backbone_signature"],
        vae_num_channels=cfg.get("vae_num_channels", 4),
        unet_input_channels=cfg.get("unet_input_channels", 4),
        source_key=cfg.get("source_key", "image"),
        target_key=cfg.get("target_key", "normal"),
        mask_key=cfg.get("mask_key", "mask"),
        timestep_sampling=cfg.get("timestep_sampling", "uniform"),
        logit_mean=cfg.get("logit_mean", 0.0),
        logit_std=cfg.get("logit_std", 1.0),
        pixel_loss_type="lpips",
        latent_loss_type=cfg.get("latent_loss_type", "l2"),
        latent_loss_weight=cfg.get("latent_loss_weight", 1.0),
        pixel_loss_weight=0.0,
        selected_timesteps=cfg.get("selected_timesteps", None),
        prob=cfg.get("prob", None),
        conditioning_images_keys=cfg.get("conditioning_images_keys", []),
        conditioning_masks_keys=cfg.get("conditioning_masks_keys", []),
        bridge_noise_sigma=cfg.get("bridge_noise_sigma", 0.005),
    )

    ckpt = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )

    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # Lightning checkpoint 里的 key 通常是 model.xxx
    model_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            model_state_dict[k.replace("model.", "", 1)] = v

    if len(model_state_dict) == 0:
        model_state_dict = state_dict

    missing, unexpected = model.load_state_dict(model_state_dict, strict=False)
    print(f"[Load ckpt] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()

    # 让 VAE / conditioner 执行和训练时一致的初始化
    model.on_fit_start(device=torch.device(device))

    return model


@torch.no_grad()
def infer_one(model, img_tensor, num_steps=1, device="cuda"):
    img_tensor = img_tensor.to(device=device, dtype=torch.bfloat16)

    # log_samples 需要 target_key 的 shape，所以这里构造一个 dummy normal
    # 你的任务推理时没有真实 photo，这里只用于提供尺寸
    batch = {
        "image": img_tensor,
        "normal": torch.zeros_like(img_tensor),
        "mask": torch.ones((img_tensor.shape[0], 1, img_tensor.shape[2], img_tensor.shape[3]),
                           device=device, dtype=torch.bfloat16),
    }

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logs = model.log_samples(
            batch,
            num_steps=[num_steps],
            max_samples=img_tensor.shape[0],
        )

    pred = logs[f"samples_{num_steps}_steps"]
    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_steps", type=int, default=1)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_lbm_model(args.config, args.ckpt, device=device)

    input_dir = Path(args.input_dir)
    img_paths = []
    for ext in IMG_EXTS:
        img_paths.extend(input_dir.glob(f"*{ext}"))
    img_paths = sorted(img_paths)

    print(f"[Infer] Found {len(img_paths)} images")

    for i, img_path in enumerate(img_paths):
        img = load_image(img_path, size=args.size)
        pred = infer_one(model, img, num_steps=args.num_steps, device=device)

        out_name = img_path.stem + f"_step{args.num_steps}.png"
        out_path = Path(args.output_dir) / out_name

        save_image(tensor_to_image(pred.float()), str(out_path))
        print(f"[{i+1}/{len(img_paths)}] saved: {out_path}")


if __name__ == "__main__":
    main()