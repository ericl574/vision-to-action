"""Fine-tune the real-data camera trajectory network on MetaDrive expert routes.

Training/validation split is by entire episode. Benchmark seeds 5000+ must
never appear here; the prepared dataset enforces this. A trajectory model is
saved only from validation ADE, never from closed-loop benchmark results.
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from train_comma1m import DriveSet
from train_comma1m_v2 import ResidualCNNGRU


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/metadrive_trajectories_v2/samples.csv"))
    parser.add_argument("--pretrained", type=Path, default=Path("checkpoints/comma1m_pipeline/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/metadrive_trajectory_v2"))
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output directory already exists; choose a new run name")
    random.seed(27)
    np.random.seed(27)
    torch.manual_seed(27)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    with args.data.open(newline="") as file:
        rows = list(csv.DictReader(file))
    ids = sorted({row["segment"] for row in rows})
    if len(ids) < 5 or any(int(segment.removeprefix("episode_")) >= 5000 for segment in ids):
        raise RuntimeError("Need at least five training-seed episodes and no benchmark seeds")
    random.shuffle(ids)
    val_ids = set(ids[:max(1, round(len(ids) * 0.2))])
    train_rows = [row for row in rows if row["segment"] not in val_ids]
    val_rows = [row for row in rows if row["segment"] in val_ids]
    device = "mps" if args.device == "auto" and torch.backends.mps.is_available() else "cpu" if args.device == "auto" else args.device
    model = ResidualCNNGRU().to(device)
    pretrained = torch.load(args.pretrained, map_location="cpu", weights_only=False)
    model.load_state_dict(pretrained["model_state_dict"])
    train_loader = DataLoader(DriveSet(train_rows), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(DriveSet(val_rows), batch_size=args.batch_size, shuffle=False, num_workers=0)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "split.json").write_text(json.dumps(dict(
        train=sorted(set(ids) - val_ids), validation=sorted(val_ids),
        pretrained=str(args.pretrained), data=str(args.data),
    ), indent=2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_ade, stale, history = float("inf"), 0, []
    print(f"Train {len(train_rows)} samples / {len(ids)-len(val_ids)} routes; "
          f"validation {len(val_rows)} / {len(val_ids)} routes; device={device}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for images, speed, target in train_loader:
            images, speed, target = images.to(device), speed.to(device), target.to(device)
            prediction, _ = model(images, speed)
            loss = nn.functional.smooth_l1_loss(prediction, target, beta=2.0)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        pred_sum = base_sum = fde_sum = base_fde_sum = count = 0.0
        with torch.inference_mode():
            for images, speed, target in val_loader:
                prediction, baseline = model(images.to(device), speed.to(device))
                truth = target.to(device)
                pred_error = torch.linalg.vector_norm(prediction - truth, dim=-1)
                base_error = torch.linalg.vector_norm(baseline - truth, dim=-1)
                pred_sum += float(pred_error.sum())
                base_sum += float(base_error.sum())
                fde_sum += float(pred_error[:, -1].sum())
                base_fde_sum += float(base_error[:, -1].sum())
                count += len(images)
        ade, fde = pred_sum / (20 * count), fde_sum / count
        record = dict(epoch=epoch, train_loss=float(np.mean(losses)), ADE_m=ade, FDE_m=fde,
                      baseline_ADE_m=base_sum / (20 * count), baseline_FDE_m=base_fde_sum / count)
        history.append(record)
        payload = dict(model_state_dict=model.state_dict(), epoch=epoch, val_ade=ade,
                       config=dict(type="residual_cnn_gru_trajectory", source="MetaDrive expert fine-tune",
                                   history_frames=4, future_waypoints=20, speed_normalization=30.0))
        torch.save(payload, args.output / "last.pt")
        if ade < best_ade:
            best_ade, stale = ade, 0
            torch.save(payload, args.output / "best.pt")
        else:
            stale += 1
        with (args.output / "history.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(record))
            writer.writeheader()
            writer.writerows(history)
        print(f"epoch={epoch} train={record['train_loss']:.3f} ADE={ade:.3f}m FDE={fde:.3f}m "
              f"constant_velocity_ADE={record['baseline_ADE_m']:.3f}m", flush=True)
        if stale >= args.patience:
            break
    print(f"Best validation ADE {best_ade:.3f}m; saved {args.output / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
