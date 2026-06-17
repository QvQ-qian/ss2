import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def sorted_images(folder):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [p for p in Path(folder).iterdir() if p.suffix.lower() in exts]

    def key_fn(p):
        stem = p.stem
        return int(stem) if stem.isdigit() else stem

    return sorted(files, key=key_fn)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="GT photo folder, e.g. train_B")
    parser.add_argument("--output_dir", required=True, help="output parse label folder")
    parser.add_argument("--decp_root", required=True, help="DECP repo path")
    parser.add_argument(
        "--bisenet_ckpt",
        required=True,
        help="pretrained_models/face_parsing_bisenet.pth",
    )
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    sys.path.insert(0, args.decp_root)
    from utils.model import BiSeNet

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    net = BiSeNet(n_classes=args.num_classes).to(device)
    state_dict = torch.load(args.bisenet_ckpt, map_location=device)
    net.load_state_dict(state_dict)
    net.eval()

    tfm = transforms.Compose(
        [
            transforms.Resize((args.size, args.size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    img_paths = sorted_images(args.input_dir)
    print(f"[FaceParsing] found {len(img_paths)} images")

    for idx, img_path in enumerate(img_paths, 1):
        img = Image.open(img_path).convert("RGB")
        x = tfm(img).unsqueeze(0).to(device)

        logits = net(x)[0]  # [1, 19, H, W]
        label = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)

        out_path = Path(args.output_dir) / f"{img_path.stem}.png"
        Image.fromarray(label, mode="L").save(out_path)

        if idx % 100 == 0:
            print(f"[FaceParsing] {idx}/{len(img_paths)}")

    print(f"[FaceParsing] saved to {args.output_dir}")


if __name__ == "__main__":
    main()