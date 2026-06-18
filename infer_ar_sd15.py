import os

# 必须放在任何 diffusers / transformers 相关 import 之前
# 避免 transformers 误触发 TensorFlow -> protobuf 的导入错误
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import argparse
from contextlib import nullcontext
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
    tfm = transforms.Compose(
        [
            transforms.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    return tfm(img).unsqueeze(0)


def tensor_to_image(x):
    # [-1, 1] -> [0, 1]
    return (x.clamp(-1, 1) + 1.0) / 2.0


def list_images(input_dir):
    input_dir = Path(input_dir)
    img_paths = []
    for ext in IMG_EXTS:
        img_paths.extend(input_dir.glob(f"*{ext}"))
        img_paths.extend(input_dir.glob(f"*{ext.upper()}"))
    return sorted(list(set(img_paths)))


def _strip_lightning_prefix(state_dict):
    """
    Lightning checkpoint keys usually start with 'model.'.
    Convert:
        model.denoiser.xxx -> denoiser.xxx
        model.vae.xxx      -> vae.xxx
    """
    model_state_dict = {}

    for k, v in state_dict.items():
        if k.startswith("model."):
            model_state_dict[k.replace("model.", "", 1)] = v

    if len(model_state_dict) == 0:
        model_state_dict = state_dict

    return model_state_dict


def load_lbm_model(config_path, ckpt_path, device="cuda"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # 推理阶段不需要 LPIPS / ID / local edge loss，所以把这些 loss 权重置 0。
    # 但模型结构相关开关必须和训练一致，尤其是 bidirectional / direction_aware。
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
        selected_timesteps=cfg.get("selected_timesteps", None),
        prob=cfg.get("prob", None),
        conditioning_images_keys=cfg.get("conditioning_images_keys", []),
        conditioning_masks_keys=cfg.get("conditioning_masks_keys", []),
        bridge_noise_sigma=cfg.get("bridge_noise_sigma", 0.005),

        # ---------- losses: inference does not need image-level losses ----------
        pixel_loss_type=cfg.get("pixel_loss_type", "lpips"),
        latent_loss_type=cfg.get("latent_loss_type", "l2"),
        latent_loss_weight=cfg.get("latent_loss_weight", 1.0),
        pixel_loss_weight=0.0,
        id_loss_weight=0.0,
        local_edge_loss_weight=0.0,

        # ---------- Bi-LBM: must match training ----------
        bidirectional=cfg.get("bidirectional", False),
        bidirectional_mode=cfg.get("bidirectional_mode", "none"),
        direction_aware=cfg.get("direction_aware", False),
        num_directions=cfg.get("num_directions", 2),
        direction_embed_init=cfg.get("direction_embed_init", 0.0),
        reverse_loss_weight=cfg.get("reverse_loss_weight", 0.5),
        reverse_use_pixel_loss=cfg.get("reverse_use_pixel_loss", False),
        reverse_use_id_loss=cfg.get("reverse_use_id_loss", False),
        reverse_use_local_edge_loss=cfg.get("reverse_use_local_edge_loss", False),
        eval_directions=cfg.get("eval_directions", ["s2p"]),
        eval_save_p2s_images=cfg.get("eval_save_p2s_images", False),

        # ---------- Bridge-MAAM: must match training if enabled ----------
        use_bridge_maam=cfg.get("use_bridge_maam", False),
        bridge_maam_mode=cfg.get("bridge_maam_mode", "residual"),
        bridge_maam_levels=cfg.get("bridge_maam_levels", None),
        bridge_maam_attn_type=cfg.get("bridge_maam_attn_type", "scsa"),
        bridge_maam_alpha_init=cfg.get("bridge_maam_alpha_init", 0.01),
        bridge_maam_attn_bias_init=cfg.get("bridge_maam_attn_bias_init", 2.0),
        bridge_maam_use_timestep=cfg.get("bridge_maam_use_timestep", True),
        bridge_maam_zero_init_timestep=cfg.get("bridge_maam_zero_init_timestep", True),
        bridge_maam_scsa_groups=cfg.get("bridge_maam_scsa_groups", 4),
        bridge_maam_scsa_kernels=cfg.get("bridge_maam_scsa_kernels", None),
        bridge_maam_scsa_pool_size=cfg.get("bridge_maam_scsa_pool_size", 7),

        # ---------- VAE skip: must match training if enabled ----------
        use_vae_skip=cfg.get("use_vae_skip", False),
        vae_skip_zero_init=cfg.get("vae_skip_zero_init", True),
        vae_skip_gamma=cfg.get("vae_skip_gamma", 1.0),

        # ---------- Face adapter: current Bi-LBM v1 normally keeps it False ----------
        use_face_adapter=cfg.get("use_face_adapter", False),
        parse_key=cfg.get("parse_key", "parse"),
        parse_num_classes=cfg.get("parse_num_classes", 19),
        parse_adapter_scale=cfg.get("parse_adapter_scale", 1.0),
        parse_adapter_condition_dropout=cfg.get("parse_adapter_condition_dropout", 0.0),
        parse_adapter_include_mid=cfg.get("parse_adapter_include_mid", True),
        parse_adapter_zero_init=cfg.get("parse_adapter_zero_init", True),
        parse_adapter_use_scale_gates=cfg.get("parse_adapter_use_scale_gates", True),
        parse_adapter_gate_init=cfg.get("parse_adapter_gate_init", 1.0),
        use_sketch_face_adapter=cfg.get("use_sketch_face_adapter", False),
        sketch_key=cfg.get("sketch_key", None),
        sketch_in_channels=cfg.get("sketch_in_channels", 3),
        use_coarse_face_adapter=cfg.get("use_coarse_face_adapter", False),
        coarse_face_key=cfg.get("coarse_face_key", None),
        coarse_in_channels=cfg.get("coarse_in_channels", 3),
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

    model_state_dict = _strip_lightning_prefix(state_dict)

    missing, unexpected = model.load_state_dict(model_state_dict, strict=False)
    print(f"[Load ckpt] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

    if len(missing) > 0:
        print("[Load ckpt] first 20 missing keys:")
        for k in missing[:20]:
            print("  ", k)

    if len(unexpected) > 0:
        print("[Load ckpt] first 20 unexpected keys:")
        for k in unexpected[:20]:
            print("  ", k)

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = model.to(device=device, dtype=dtype)
    model.eval()

    # 让 VAE / conditioner 执行和训练时一致的初始化
    model.on_fit_start(device=torch.device(device))

    # 简单检查 direction embedding 是否真的启用
    denoiser = getattr(model, "denoiser", None)
    if getattr(model, "direction_aware", False):
        has_dir = hasattr(denoiser, "enable_direction_embedding") or hasattr(
            denoiser, "use_direction_embedding"
        )
        print(
            "[Bi-LBM] direction_aware=True, "
            f"denoiser_has_direction_support={has_dir}, "
            f"use_direction_embedding={getattr(denoiser, 'use_direction_embedding', None)}"
        )

    return model


def build_infer_batch(img_tensor, direction, device):
    """
    direction:
        s2p: image/sketch  -> normal/photo
        p2s: normal/photo -> image/sketch
    """
    img_tensor = img_tensor.to(device=device)

    if direction == "s2p":
        # 输入是 sketch，放到 image 端
        batch = {
            "image": img_tensor,
            "normal": torch.zeros_like(img_tensor),
            "mask": torch.ones(
                (img_tensor.shape[0], 1, img_tensor.shape[2], img_tensor.shape[3]),
                device=device,
                dtype=img_tensor.dtype,
            ),
        }
        src_pixels = batch["image"]

    elif direction == "p2s":
        # 输入是 photo，放到 normal 端
        batch = {
            "image": torch.zeros_like(img_tensor),
            "normal": img_tensor,
            "mask": torch.ones(
                (img_tensor.shape[0], 1, img_tensor.shape[2], img_tensor.shape[3]),
                device=device,
                dtype=img_tensor.dtype,
            ),
        }
        src_pixels = batch["normal"]

    else:
        raise ValueError(f"Unknown direction: {direction}")

    return batch, src_pixels


@torch.no_grad()
def infer_one(model, img_tensor, num_steps=1, device="cuda", direction="s2p"):
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    img_tensor = img_tensor.to(device=device, dtype=dtype)

    batch, src_pixels = build_infer_batch(
        img_tensor=img_tensor,
        direction=direction,
        device=device,
    )

    if device.startswith("cuda"):
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast_ctx = nullcontext()

    with autocast_ctx:
        if model.vae is not None:
            z_src = model.vae.encode(src_pixels)
        else:
            z_src = src_pixels

        pred = model.sample(
            z_src,
            num_steps=num_steps,
            conditioner_inputs=batch,
            max_samples=img_tensor.shape[0],
            verbose=False,
            direction=direction,
        )

    return pred


def run_folder(
    model,
    input_dir,
    output_dir,
    num_steps=1,
    size=256,
    device="cuda",
    direction="s2p",
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_paths = list_images(input_dir)
    print(f"[Infer:{direction}] input_dir={input_dir}")
    print(f"[Infer:{direction}] output_dir={output_dir}")
    print(f"[Infer:{direction}] Found {len(img_paths)} images")

    for i, img_path in enumerate(img_paths):
        img = load_image(img_path, size=size)
        pred = infer_one(
            model,
            img,
            num_steps=num_steps,
            device=device,
            direction=direction,
        )

        out_name = img_path.stem + f"_{direction}_step{num_steps}.png"
        out_path = output_dir / out_name

        save_image(tensor_to_image(pred.float()), str(out_path))
        print(f"[{direction}][{i + 1}/{len(img_paths)}] saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)

    # 单方向推理时使用 input_dir / output_dir
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

    # 双方向推理时推荐显式给 test_A / test_B
    parser.add_argument("--input_dir_s2p", type=str, default=None)
    parser.add_argument("--input_dir_p2s", type=str, default=None)
    parser.add_argument("--output_dir_s2p", type=str, default=None)
    parser.add_argument("--output_dir_p2s", type=str, default=None)

    parser.add_argument("--num_steps", type=int, default=1)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument(
        "--direction",
        type=str,
        default="s2p",
        choices=["s2p", "p2s", "both"],
        help="s2p: sketch->photo, p2s: photo->sketch, both: run both directions",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    model = load_lbm_model(args.config, args.ckpt, device=device)

    if args.direction in ["s2p", "p2s"]:
        if args.input_dir is None or args.output_dir is None:
            raise ValueError(
                "For --direction s2p or p2s, please provide --input_dir and --output_dir."
            )

        run_folder(
            model=model,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            num_steps=args.num_steps,
            size=args.size,
            device=device,
            direction=args.direction,
        )

    elif args.direction == "both":
        if args.input_dir_s2p is None:
            raise ValueError("For --direction both, please provide --input_dir_s2p.")
        if args.input_dir_p2s is None:
            raise ValueError("For --direction both, please provide --input_dir_p2s.")

        if args.output_dir_s2p is None or args.output_dir_p2s is None:
            if args.output_dir is None:
                raise ValueError(
                    "For --direction both, provide either --output_dir, "
                    "or both --output_dir_s2p and --output_dir_p2s."
                )

            base_out = Path(args.output_dir)
            output_dir_s2p = base_out / f"s2p_step{args.num_steps}"
            output_dir_p2s = base_out / f"p2s_step{args.num_steps}"
        else:
            output_dir_s2p = Path(args.output_dir_s2p)
            output_dir_p2s = Path(args.output_dir_p2s)

        run_folder(
            model=model,
            input_dir=args.input_dir_s2p,
            output_dir=output_dir_s2p,
            num_steps=args.num_steps,
            size=args.size,
            device=device,
            direction="s2p",
        )

        run_folder(
            model=model,
            input_dir=args.input_dir_p2s,
            output_dir=output_dir_p2s,
            num_steps=args.num_steps,
            size=args.size,
            device=device,
            direction="p2s",
        )


if __name__ == "__main__":
    main()