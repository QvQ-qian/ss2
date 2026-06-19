import argparse
import json
import os
import re
import time
from pathlib import Path

import swanlab


def infer_direction(json_path: str):
    norm_path = os.path.normpath(json_path)
    parts = norm_path.split(os.sep)

    if "s2p" in parts:
        return "s2p"
    if "p2s" in parts:
        return "p2s"

    base = os.path.basename(json_path)
    if "_s2p_" in base:
        return "s2p"
    if "_p2s_" in base:
        return "p2s"

    return None


def infer_global_step(json_path: str):
    base = os.path.basename(json_path)

    m = re.search(r"global_step_(\d+)", base)
    if m is not None:
        return int(m.group(1))

    # fallback: parent dir may contain global_step_xxxxxxxx
    for part in Path(json_path).parts[::-1]:
        m = re.search(r"global_step_(\d+)", part)
        if m is not None:
            return int(m.group(1))

    return None


def prefix_metrics(raw_metrics: dict, direction: str):
    metrics = {}

    for k, v in raw_metrics.items():
        if not isinstance(v, (int, float)):
            continue

        v = float(v)

        if direction in ["s2p", "p2s"]:
            prefix = f"external/{direction}/"

            if k.startswith(prefix):
                new_key = k
            elif k.startswith("external/"):
                new_key = k.replace("external/", prefix, 1)
            else:
                new_key = prefix + k
        else:
            new_key = k

        metrics[new_key] = v

    return metrics


def try_load_json(json_path: str):
    try:
        with open(json_path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        # 文件可能还在写入，watch 模式下下轮再读
        return None
    except Exception as e:
        print(f"[UploadExternalMetrics] failed to read {json_path}: {repr(e)}")
        return None


def collect_json_files(metrics_root: str):
    root = Path(metrics_root)
    if not root.exists():
        print(f"[UploadExternalMetrics] metrics_root not found: {metrics_root}")
        return []

    return sorted(str(p) for p in root.rglob("*.json"))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--metrics_root", type=str, required=True)

    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("--workspace", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--resume", type=str, default="must", choices=["must", "allow"])

    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--mode", type=str, default="online")

    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll_interval", type=int, default=30)
    parser.add_argument("--max_wait_seconds", type=int, default=0)

    args = parser.parse_args()

    run = swanlab.init(
        project=args.project,
        workspace=args.workspace,
        experiment_name=args.experiment_name,
        id=args.run_id,
        resume=args.resume,
        logdir=args.logdir,
        mode=args.mode,
    )

    print(f"[UploadExternalMetrics] resume SwanLab run id={args.run_id}")
    print(f"[UploadExternalMetrics] metrics_root={args.metrics_root}")

    uploaded = set()
    start_time = time.time()

    while True:
        json_files = collect_json_files(args.metrics_root)

        num_uploaded_this_round = 0

        for json_path in json_files:
            if json_path in uploaded:
                continue

            raw_metrics = try_load_json(json_path)
            if raw_metrics is None:
                continue

            global_step = infer_global_step(json_path)
            if global_step is None:
                print(f"[UploadExternalMetrics] skip no global_step: {json_path}")
                uploaded.add(json_path)
                continue

            direction = infer_direction(json_path)
            metrics = prefix_metrics(raw_metrics, direction)

            if len(metrics) == 0:
                print(f"[UploadExternalMetrics] skip empty metrics: {json_path}")
                uploaded.add(json_path)
                continue

            swanlab.log(metrics, step=global_step)

            uploaded.add(json_path)
            num_uploaded_this_round += 1

            print(
                f"[UploadExternalMetrics] uploaded step={global_step}, "
                f"direction={direction}, file={json_path}, keys={list(metrics.keys())}"
            )

        if not args.watch:
            break

        if args.max_wait_seconds > 0:
            elapsed = time.time() - start_time
            if elapsed > args.max_wait_seconds:
                print("[UploadExternalMetrics] max_wait_seconds reached, stop watching.")
                break

        print(
            f"[UploadExternalMetrics] watch mode: uploaded_total={len(uploaded)}, "
            f"new_this_round={num_uploaded_this_round}, sleep={args.poll_interval}s"
        )
        time.sleep(args.poll_interval)

    swanlab.finish()
    print(f"[UploadExternalMetrics] done. uploaded={len(uploaded)} json files.")


if __name__ == "__main__":
    main()