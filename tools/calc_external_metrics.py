# tools/calc_external_metrics.py

import os
import sys
import json
import argparse
from pathlib import Path

# 让脚本可以 import 项目根目录下的 ALL.py 和 rank_new.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import ALL


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--gen_dir", required=True)
    parser.add_argument("--out_json", required=True)

    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--deepface_home", default="/root/shuqian/checkpoints")

    parser.add_argument("--do_image_metrics", action="store_true")
    parser.add_argument("--do_rank_metrics", action="store_true")

    parser.add_argument(
        "--local_inception_v3_path",
        default="/root/shuqian/checkpoints/inception_v3_google-0cc3c7bd.pth",
    )

    # 新增：当前计算的是几步生成结果
    parser.add_argument(
        "--step_num",
        type=int,
        default=4,
        help="Generation step number used in metric names, e.g., 1 or 4.",
    )

    args = parser.parse_args()

    os.environ["DEEPFACE_HOME"] = args.deepface_home

    step = int(args.step_num)
    metrics = {}

    print("[ExternalMetrics] start")
    print(f"[ExternalMetrics] gt_dir: {args.gt_dir}")
    print(f"[ExternalMetrics] gen_dir: {args.gen_dir}")
    print(f"[ExternalMetrics] out_json: {args.out_json}")
    print(f"[ExternalMetrics] step_num: {step}")

    if args.do_image_metrics:
        img_metrics = ALL.calculate(
            real_images=args.gt_dir,
            generated_images=args.gen_dir,
            batch_size=args.batch_size,
            device=args.device,
            local_inception_v3_path=args.local_inception_v3_path,
        )

        metrics.update(
            {
                f"external/fid_step{step}": safe_float(img_metrics.get("fid", 0)),
                f"external/mssim_step{step}": safe_float(img_metrics.get("avg_mssim", 0)),
                f"external/vif_step{step}": safe_float(img_metrics.get("avg_vif", 0)),
            }
        )

    if args.do_rank_metrics:
        try:
            import rank_new

            rank_metrics = rank_new.calculate(
                gt_dir=args.gt_dir,
                gen_dir=args.gen_dir,
            )

            for k, v in rank_metrics.items():
                if isinstance(v, (int, float)):
                    metrics[f"external/{k}_step{args.step_num}"] = float(v)

        except Exception as e:
            print(f"[ExternalMetrics] rank_new failed: {repr(e)}")
            metrics[f"external/rank_failed_step{args.step_num}"] = 1.0

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    with open(args.out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[ExternalMetrics] saved to {args.out_json}")
    print(metrics)


if __name__ == "__main__":
    main()