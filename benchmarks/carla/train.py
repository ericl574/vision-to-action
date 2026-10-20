"""Resumable training for PilotNet.

    python benchmarks/carla/train.py --data /workspace/data/expert_town10 /workspace/data/expert_town03 \
        --output /workspace/runs/pilot_v1 --epochs 15

Fine-tune (DAgger round) while keeping the original validation routes:

    python benchmarks/carla/train.py --data <expert roots> <dagger roots> --output /workspace/runs/pilot_v2 \
        --initial-checkpoint /workspace/runs/pilot_v1/best.pt --split-file /workspace/runs/pilot_v1/split.json \
        --epochs 8 --lr 1e-4
"""

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.carla.dataset import build_samples, create_dataset, describe, sample_weights  # noqa: E402
from benchmarks.carla.model import DIST_NORM_M, SPEED_NORM_MPS, create_network  # noqa: E402

LOSS_WEIGHTS = {"steer": 1.0, "speed": 1.0, "tl": 0.5, "hazard": 0.5, "dist": 0.2}


def compute_losses(out, batch, F):
    steer = F.l1_loss(out["steer"], batch["steer"])
    speed = F.l1_loss(out["target_speed"] / SPEED_NORM_MPS, batch["target_speed"] / SPEED_NORM_MPS)
    tl = F.cross_entropy(out["tl_logits"], batch["tl_state"])
    hazard = F.binary_cross_entropy_with_logits(out["hazard_logit"], batch["hazard"])
    mask = batch["hazard"]
    dist = (F.l1_loss(out["hazard_dist"] / DIST_NORM_M, batch["hazard_dist"] / DIST_NORM_M, reduction="none") * mask).sum() / mask.sum().clamp(min=1)
    total = (LOSS_WEIGHTS["steer"] * steer + LOSS_WEIGHTS["speed"] * speed + LOSS_WEIGHTS["tl"] * tl
             + LOSS_WEIGHTS["hazard"] * hazard + LOSS_WEIGHTS["dist"] * dist)
    return total, {"steer": steer, "speed": speed, "tl": tl, "hazard": hazard, "dist": dist}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", required=True, help="route roots produced by collect.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--history", type=int, default=3)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--backbone", default="resnet34")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--initial-checkpoint", default=None)
    parser.add_argument("--resume", action="store_true", help="continue from <output>/last.pt")
    parser.add_argument("--frame-step", type=int, default=1, help="use every k-th frame as a sample")
    parser.add_argument("--max-steps-per-epoch", type=int, default=0, help="0 = full pass; >0 caps an epoch")
    parser.add_argument("--seed", type=int, default=27)
    args = parser.parse_args()

    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, WeightedRandomSampler

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_file = args.split_file or (output / "split.json" if args.resume else None)
    train_rows, val_rows, split = build_samples(args.data, args.history, args.stride, args.val_frac, split_file,
                                                args.seed, args.frame_step)
    with open(output / "split.json", "w") as handle:
        json.dump(split, handle, indent=1)
    stats = {"train": describe(train_rows), "val": describe(val_rows)}
    with open(output / "data_stats.json", "w") as handle:
        json.dump(stats, handle, indent=1)
    print("train:", json.dumps(stats["train"]))
    print("val:  ", json.dumps(stats["val"]))

    train_set = create_dataset(train_rows, args.width, args.height, augment=True)
    val_set = create_dataset(val_rows, args.width, args.height, augment=False)
    weights = sample_weights(train_rows)
    epoch_samples = args.max_steps_per_epoch * args.batch_size if args.max_steps_per_epoch else len(train_rows)
    sampler = WeightedRandomSampler(torch.as_tensor(weights), num_samples=epoch_samples, replacement=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
                              pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                            pin_memory=True, persistent_workers=args.workers > 0)

    model = create_network(args.history, pretrained=not args.no_pretrained, backbone=args.backbone).to(device)
    config = dict(model.config, width=args.width, height=args.height, stride=args.stride)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup = max(1, len(train_loader) // 2)

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch, best_val = 0, float("inf")

    if args.initial_checkpoint:
        payload = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        print(f"initialised from {args.initial_checkpoint} (epoch {payload.get('epoch')})")
    if args.resume and (output / "last.pt").exists():
        payload = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        start_epoch = payload["epoch"] + 1
        best_val = payload.get("best_val", best_val)
        print(f"resumed from epoch {payload['epoch']}")

    history_path = output / "history.csv"
    if not history_path.exists() or not args.resume:
        with open(history_path, "w", newline="") as handle:
            csv.writer(handle).writerow(["epoch", "train_loss", "val_loss", "val_steer_mae", "val_speed_mae",
                                         "val_tl_acc", "val_hazard_acc", "lr", "seconds"])

    def save(path, epoch, val):
        torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(), "epoch": epoch, "config": config,
                    "val": val, "best_val": best_val, "args": vars(args)}, path)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        start = time.time()
        running, count = 0.0, 0
        for batch in train_loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(batch["images"], batch["speed"], batch["command"])
                loss, _ = compute_losses(out, batch, F)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += float(loss)
            count += 1
            if count % 50 == 0:
                print(f"epoch {epoch} step {count}/{len(train_loader)} loss {running / count:.4f}", flush=True)

        model.eval()
        totals = {"loss": 0.0, "steer": 0.0, "speed": 0.0, "tl_correct": 0, "hazard_correct": 0, "n": 0}
        with torch.no_grad():
            for batch in val_loader:
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                out = model(batch["images"], batch["speed"], batch["command"])
                loss, parts = compute_losses(out, batch, F)
                n = batch["steer"].shape[0]
                totals["loss"] += float(loss) * n
                totals["steer"] += float((out["steer"] - batch["steer"]).abs().sum())
                totals["speed"] += float((out["target_speed"] - batch["target_speed"]).abs().sum())
                totals["tl_correct"] += int((out["tl_logits"].argmax(1) == batch["tl_state"]).sum())
                totals["hazard_correct"] += int(((out["hazard_logit"] > 0).float() == batch["hazard"]).sum())
                totals["n"] += n
        n = max(1, totals["n"])
        val = {"loss": totals["loss"] / n, "steer_mae": totals["steer"] / n, "speed_mae": totals["speed"] / n,
               "tl_acc": totals["tl_correct"] / n, "hazard_acc": totals["hazard_correct"] / n}
        control_val = val["steer_mae"] + val["speed_mae"] / SPEED_NORM_MPS
        seconds = time.time() - start
        print(f"epoch {epoch}: train {running / max(1, count):.4f} | val loss {val['loss']:.4f} "
              f"steer MAE {val['steer_mae']:.4f} speed MAE {val['speed_mae']:.3f} m/s "
              f"tl acc {val['tl_acc']:.3f} hazard acc {val['hazard_acc']:.3f} | {seconds:.0f}s", flush=True)
        with open(history_path, "a", newline="") as handle:
            csv.writer(handle).writerow([epoch, running / max(1, count), val["loss"], val["steer_mae"],
                                         val["speed_mae"], val["tl_acc"], val["hazard_acc"],
                                         scheduler.get_last_lr()[0], round(seconds)])
        if control_val < best_val:
            best_val = control_val
            save(output / "best.pt", epoch, val)
            print(f"  new best (control val {control_val:.4f}) -> {output / 'best.pt'}")
        save(output / "last.pt", epoch, val)


if __name__ == "__main__":
    main()
