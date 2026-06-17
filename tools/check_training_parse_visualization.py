#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check whether face parsing maps are loaded and prepared correctly for the LBM Face Parsing Adapter.

Run from the root of your ss1 repo, for example:

python tools/check_training_parse_visualization.py \
  --shards "/path/to/ar_train-%06d.tar" \
  --out_dir debug_parse_check \
  --num_samples 8 \
  --simulate_mapper totensor_rescale \
  --run_adapter

Why this script exists:
- make_ar_lbm_webdataset.py stores parse.png as a single-channel label map with values 0..18.
- FaceConditionalAdapter expects parse as one-hot tensor [B, 19, H, W].
- LBMModel._get_face_adapter_residuals currently sends batch[parse_key] directly into the adapter,
  so this script helps reveal whether the parse tensor is already one-hot or accidentally scaled.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F

try:
    import webdataset as wds
except Exception as exc:
    raise RuntimeError("Please install webdataset first: pip install webdataset") from exc


# 19-class CelebAMask-HQ/BiSeNet-like palette. Exact colors are only for visualization.
PALETTE = np.array(
    [
        [0, 0, 0],        # 0 background
        [204, 0, 0],      # 1 skin/face
        [76, 153, 0],
        [204, 204, 0],
        [51, 51, 255],
        [204, 0, 204],
        [0, 255, 255],
        [255, 204, 204],
        [102, 51, 0],
        [255, 0, 0],
        [102, 204, 0],
        [255, 255, 0],
        [0, 0, 153],
        [0, 0, 204],
        [255, 51, 153],
        [0, 204, 204],
        [0, 51, 0],
        [255, 153, 51],
        [0, 204, 0],
    ],
    dtype=np.uint8,
)


def pil_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB")


def pil_to_tensor01(img: Image.Image) -> torch.Tensor:
    """Mimic torchvision.transforms.ToTensor for PIL images."""
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    ten = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float() / 255.0
    return ten


def label_to_color(label: np.ndarray, num_classes: int = 19) -> Image.Image:
    label = label.astype(np.int64)
    label = np.clip(label, 0, num_classes - 1)
    color = PALETTE[label]
    return Image.fromarray(color, mode="RGB")


def overlay_parse_on_image(image: Image.Image, label: np.ndarray, alpha: float = 0.45) -> Image.Image:
    base = pil_rgb(image).resize((label.shape[1], label.shape[0]), Image.BICUBIC)
    color = label_to_color(label)
    return Image.blend(base, color, alpha=alpha)


def image_with_title(img: Image.Image, title: str, width: int = 256, title_h: int = 28) -> Image.Image:
    img = pil_rgb(img).resize((width, width), Image.NEAREST)
    canvas = Image.new("RGB", (width, width + title_h), (255, 255, 255))
    canvas.paste(img, (0, title_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 7), title, fill=(0, 0, 0))
    return canvas


def make_grid(items, cols: int = 3) -> Image.Image:
    w, h = items[0].size
    rows = int(np.ceil(len(items) / cols))
    grid = Image.new("RGB", (cols * w, rows * h), (255, 255, 255))
    for i, img in enumerate(items):
        x = (i % cols) * w
        y = (i // cols) * h
        grid.paste(img, (x, y))
    return grid


def summarize_numpy_label(label: np.ndarray, name: str, num_classes: int) -> None:
    unique, counts = np.unique(label, return_counts=True)
    print(f"\n[{name}]")
    print(f"  shape={label.shape}, dtype={label.dtype}, min={label.min()}, max={label.max()}")
    print(f"  unique={unique.tolist()}")
    if label.ndim == 2:
        bad = unique[(unique < 0) | (unique >= num_classes)]
        if len(bad) > 0:
            print(f"  [BAD] Found labels outside [0,{num_classes - 1}]: {bad.tolist()}")
        if len(unique) > num_classes:
            print(f"  [BAD] Too many unique values for a label map. It may have been interpolated or saved as visualization.")
        count_dict = {int(u): int(c) for u, c in zip(unique, counts)}
        print(f"  class_counts={count_dict}")


def summarize_tensor(t: torch.Tensor, name: str, max_unique: int = 30) -> None:
    td = t.detach().cpu()
    print(f"\n[{name}]")
    print(f"  shape={tuple(td.shape)}, dtype={td.dtype}, min={td.min().item():.8f}, max={td.max().item():.8f}")
    flat = td.flatten()
    unique = torch.unique(flat)
    if unique.numel() <= max_unique:
        vals = [float(x) for x in unique.tolist()]
        print(f"  unique={vals}")
    else:
        print(f"  unique_count={unique.numel()} | first_{max_unique}={unique[:max_unique].tolist()}")


def simulate_mapper_tensor(label: np.ndarray, mode: str) -> torch.Tensor:
    """
    Return a batch-like tensor [1,C,H,W] to emulate possible training mapper outputs.

    label:
      Keeps labels as 0..18 in [B,1,H,W]. This is OK only if later converted to one-hot.
    totensor:
      Mimics torchvision ToTensor on parse.png, causing labels 0..18 to become 0..0.070588.
    totensor_rescale:
      Mimics ToTensor followed by RescaleMapper, causing labels to become about -1..-0.8588.
    onehot:
      Correct final format expected by FaceConditionalAdapter: [B,19,H,W].
    """
    x_label = torch.from_numpy(label.astype(np.int64))

    if mode == "label":
        return x_label.unsqueeze(0).unsqueeze(0).float()

    if mode == "totensor":
        return x_label.unsqueeze(0).unsqueeze(0).float() / 255.0

    if mode == "totensor_rescale":
        x = x_label.unsqueeze(0).unsqueeze(0).float() / 255.0
        return 2.0 * x - 1.0

    if mode == "onehot":
        return F.one_hot(x_label.clamp(0, 18), num_classes=19).permute(2, 0, 1).unsqueeze(0).float()

    raise ValueError(f"Unknown simulate_mapper mode: {mode}")


def prepare_parse_for_adapter(parse: torch.Tensor, num_classes: int = 19) -> torch.Tensor:
    """
    Same safe conversion you should place before FaceConditionalAdapter.

    Accepts:
      [B,19,H,W] one-hot or soft one-hot
      [B,1,H,W] label map, possibly raw 0..18, ToTensor-scaled 0..18/255, or RescaleMapper-scaled [-1, -0.8588]
      [B,H,W] label map

    Returns:
      [B,19,H,W] float32 one-hot
    """
    if parse.ndim == 4 and parse.shape[1] == num_classes:
        out = parse.float()
        if out.min() < -1e-4 or out.max() > 1 + 1e-4:
            raise ValueError(
                f"Tensor looks one-hot by channel count, but values are invalid: "
                f"min={out.min().item()}, max={out.max().item()}"
            )
        return out

    if parse.ndim == 4 and parse.shape[1] == 1:
        parse = parse[:, 0]

    if parse.ndim != 3:
        raise ValueError(f"Unsupported parse shape: {tuple(parse.shape)}")

    pmin = float(parse.min().item())
    pmax = float(parse.max().item())
    x = parse.float()

    # Accidentally sent through ToTensor: original labels 0..18 become 0..0.070588.
    if pmin >= 0.0 and pmax <= 1.0:
        x = torch.round(x * 255.0)

    # Accidentally sent through ToTensor + RescaleMapper: original labels become [-1, -0.8588].
    elif pmin < 0.0:
        x = torch.round(((x + 1.0) / 2.0) * 255.0)

    # Else assume raw labels 0..18.
    else:
        x = torch.round(x)

    x = x.long()
    if x.min() < 0 or x.max() >= num_classes:
        raise ValueError(
            f"Invalid parse label range after conversion: "
            f"shape={tuple(x.shape)}, min={x.min().item()}, max={x.max().item()}, num_classes={num_classes}"
        )

    out = F.one_hot(x, num_classes=num_classes).permute(0, 3, 1, 2).contiguous().float()
    return out


def check_onehot(parse_oh: torch.Tensor, num_classes: int = 19) -> None:
    summarize_tensor(parse_oh, "adapter_ready_parse_onehot")
    ch_sum = parse_oh.sum(dim=1)
    print(f"  channel_sum min={ch_sum.min().item():.6f}, max={ch_sum.max().item():.6f}")
    label = parse_oh.argmax(dim=1)
    bincount = torch.bincount(label.flatten().cpu(), minlength=num_classes)
    print(f"  argmax class_counts={bincount.tolist()}")

    if parse_oh.ndim != 4 or parse_oh.shape[1] != num_classes:
        print(f"  [BAD] Expected [B,{num_classes},H,W], got {tuple(parse_oh.shape)}")
    if not torch.allclose(ch_sum, torch.ones_like(ch_sum), atol=1e-4):
        print("  [BAD] One-hot channel sum is not 1 everywhere.")


def maybe_run_adapter(parse_oh: torch.Tensor, repo_root: Path, include_mid: bool) -> None:
    src_path = repo_root / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))

    try:
        from lbm.models.face_condition_adapter import FaceConditionalAdapter
    except Exception as exc:
        print("\n[adapter_forward] skipped: could not import lbm.models.face_condition_adapter")
        print(f"  error={repr(exc)}")
        return

    adapter = FaceConditionalAdapter(
        parse_in_channels=parse_oh.shape[1],
        use_parse=True,
        use_coarse=False,
        include_mid=include_mid,
        zero_init=True,
        condition_dropout=0.0,
    ).eval()

    with torch.no_grad():
        out = adapter(parse=parse_oh, coarse_face=None)

    print("\n[adapter_forward]")
    for i, x in enumerate(out["down"]):
        print(f"  down[{i}] shape={tuple(x.shape)}, norm={x.float().norm().item():.8f}")
    if out.get("mid", None) is not None:
        print(f"  mid shape={tuple(out['mid'].shape)}, norm={out['mid'].float().norm().item():.8f}")
    print("  note: norm can be 0 with zero_init=True; this only verifies shape/path, not training usefulness.")


def get_sample(shards: str, index: int) -> Dict:
    dataset = wds.WebDataset(shards).decode("pil")
    for i, sample in enumerate(dataset):
        if i == index:
            return sample
    raise IndexError(f"sample_index={index} out of range")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=str, required=True, help="WebDataset tar path or shard pattern.")
    parser.add_argument("--out_dir", type=str, default="debug_parse_check")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--source_key", type=str, default="jpg")
    parser.add_argument("--target_key", type=str, default="normal_aligned.png")
    parser.add_argument("--parse_key", type=str, default="parse.png")
    parser.add_argument(
        "--simulate_mapper",
        type=str,
        default="totensor_rescale",
        choices=["label", "totensor", "totensor_rescale", "onehot"],
        help=(
            "Emulate the tensor that may enter LBMModel. "
            "Use totensor_rescale if you suspect parse went through TorchvisionMapper + RescaleMapper."
        ),
    )
    parser.add_argument("--run_adapter", action="store_true", help="Also run FaceConditionalAdapter forward if repo src is importable.")
    parser.add_argument("--include_mid", action="store_true", help="Use include_mid=True when running adapter forward.")
    parser.add_argument("--repo_root", type=str, default=".")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(args.repo_root).resolve()

    print("====== Parse Loading Check ======")
    print(f"shards={args.shards}")
    print(f"simulate_mapper={args.simulate_mapper}")
    print(f"out_dir={out_dir.resolve()}")
    print("This script reads samples with webdataset.decode('pil'), same decoder style used by the repo DataPipeline.")

    for n in range(args.num_samples):
        idx = args.start_index + n
        sample = get_sample(args.shards, idx)
        key = sample.get("__key__", f"sample_{idx}")
        print(f"\n\n========== sample {idx}: {key} ==========")
        print("keys:", sorted([k for k in sample.keys() if not k.startswith("__")]))

        if args.parse_key not in sample:
            print(f"[BAD] parse_key={args.parse_key!r} not found in sample.")
            continue

        sketch = sample.get(args.source_key, None)
        photo = sample.get(args.target_key, None)
        parse_img = sample[args.parse_key]

        if sketch is None:
            print(f"[WARN] source_key={args.source_key!r} not found.")
            sketch = Image.new("RGB", (256, 256), (255, 255, 255))
        if photo is None:
            print(f"[WARN] target_key={args.target_key!r} not found.")
            photo = Image.new("RGB", (256, 256), (255, 255, 255))

        parse_label = np.asarray(parse_img)
        if parse_label.ndim == 3:
            print(f"[WARN] parse image has {parse_label.shape[-1]} channels; using first channel for label visualization.")
            parse_label = parse_label[..., 0]

        summarize_numpy_label(parse_label, "raw_parse_png_from_wds", args.num_classes)

        # Emulate possible tensor after training mappers.
        batch_parse = simulate_mapper_tensor(parse_label, args.simulate_mapper)
        summarize_tensor(batch_parse, f"simulated_batch_parse_before_LBMModel ({args.simulate_mapper})")

        # This is the correct pre-adapter format.
        parse_oh = prepare_parse_for_adapter(batch_parse, num_classes=args.num_classes)
        check_onehot(parse_oh, num_classes=args.num_classes)

        label_after = parse_oh[0].argmax(dim=0).cpu().numpy().astype(np.uint8)
        if not np.array_equal(label_after, parse_label.astype(np.uint8)):
            diff_ratio = float((label_after != parse_label).mean())
            print(f"  [WARN] label after prepare differs from raw label. diff_ratio={diff_ratio:.6f}")
        else:
            print("  label_after_prepare matches raw_parse_png exactly.")

        if args.run_adapter:
            maybe_run_adapter(parse_oh, repo_root=repo_root, include_mid=args.include_mid)

        raw_gray = Image.fromarray(np.clip(parse_label, 0, 18).astype(np.uint8) * int(255 / max(args.num_classes - 1, 1)))
        parse_color = label_to_color(parse_label, num_classes=args.num_classes)
        prepared_color = label_to_color(label_after, num_classes=args.num_classes)
        overlay = overlay_parse_on_image(sketch, parse_label)

        grid = make_grid(
            [
                image_with_title(sketch, "sketch/source"),
                image_with_title(photo, "photo/target"),
                image_with_title(raw_gray.convert("RGB"), "raw parse gray"),
                image_with_title(parse_color, "raw parse color"),
                image_with_title(overlay, "parse overlay sketch"),
                image_with_title(prepared_color, "adapter argmax color"),
            ],
            cols=3,
        )
        safe_key = str(key).replace("/", "_")
        out_path = out_dir / f"{idx:04d}_{safe_key}_parse_check.png"
        grid.save(out_path)
        print(f"saved visualization: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
