"""Train a camera-only behavioral-cloning steering baseline on expert routes."""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.metadrive.action_model import create_network


class ActionSet(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        images = []
        for path in json.loads(row["history"]):
            with Image.open(path) as image:
                rgb = image.convert("RGB").resize((192, 120), Image.Resampling.BILINEAR)
                images.append(np.asarray(rgb, dtype=np.uint8))
        frames = torch.from_numpy(np.stack(images).copy()).permute(0, 3, 1, 2).float().div_(255)
        speed = torch.tensor([float(row["speed"]) / 30.0], dtype=torch.float32)
        steering = torch.tensor(float(row["steering"]), dtype=torch.float32)
        return frames, speed, steering


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/metadrive_actions_v1/samples.csv"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/metadrive_actions_v1"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--initial-checkpoint", type=Path, default=None)
    parser.add_argument("--split-file", type=Path, default=None,
                        help="Reuse a previous episode validation split for a comparable fine-tune")
    parser.add_argument("--dagger-repeat", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output directory exists; choose a new run name")
    random.seed(27)
    np.random.seed(27)
    torch.manual_seed(27)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    with args.data.open(newline="") as file:
        rows = list(csv.DictReader(file))
    ids = sorted({row["segment"] for row in rows})
    if len(ids) < 5 or any(int(x.removeprefix("episode_")) >= 5000 for x in ids):
        raise RuntimeError("Need at least five training-seed routes; no benchmark seeds")
    if args.split_file:
        previous = json.loads(args.split_file.read_text())
        val_ids = set(previous["validation"])
        if not val_ids <= set(ids):
            raise RuntimeError("Previous validation episodes are missing from the dataset")
    else:
        random.shuffle(ids)
        val_ids = set(ids[:max(1, round(len(ids) * 0.2))])
    train_rows = [row for row in rows if row["segment"] not in val_ids]
    val_rows = [row for row in rows if row["segment"] in val_ids]
    if args.dagger_repeat < 1:
        parser.error("dagger-repeat must be >= 1")
    if args.dagger_repeat > 1:
        dagger_rows = [row for row in train_rows if "_dagger/" in row["history"]]
        train_rows.extend(dagger_rows * (args.dagger_repeat - 1))
    train_loader = DataLoader(ActionSet(train_rows), batch_size=args.batch_size, shuffle=True,
                              num_workers=2, persistent_workers=True)
    val_loader = DataLoader(ActionSet(val_rows), batch_size=args.batch_size,
                            num_workers=2, persistent_workers=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = create_network().to(device)
    if args.initial_checkpoint:
        payload = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "split.json").write_text(json.dumps(dict(
        train=sorted(set(ids) - val_ids), validation=sorted(val_ids), data=str(args.data),
        initial_checkpoint=str(args.initial_checkpoint) if args.initial_checkpoint else None,
        dagger_repeat=args.dagger_repeat,
    ), indent=2))
    best, stale, history = float("inf"), 0, []
    print(f"Train {len(train_rows)} samples/{len(ids)-len(val_ids)} routes, "
          f"validation {len(val_rows)}/{len(val_ids)} routes, device={device}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for images, speed, target in train_loader:
            images, speed, target = images.to(device), speed.to(device), target.to(device)
            prediction = model(images, speed)
            per_sample = nn.functional.smooth_l1_loss(prediction, target, beta=0.05, reduction="none")
            turn_weight = 1.0 + 3.0 * torch.clamp(torch.abs(target) / 0.2, max=1.0)
            loss = (per_sample * turn_weight).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        absolute, weighted, turns, zero_error, turn_n, count = 0.0, 0.0, 0.0, 0.0, 0, 0
        with torch.inference_mode():
            for images, speed, target in val_loader:
                prediction = model(images.to(device), speed.to(device))
                truth = target.to(device)
                error = torch.abs(prediction - truth)
                turn_weight = 1.0 + 3.0 * torch.clamp(torch.abs(truth) / 0.2, max=1.0)
                absolute += float(error.sum())
                zero_error += float(torch.abs(truth).sum())
                weighted += float((error * turn_weight).sum())
                mask = torch.abs(truth) >= 0.1
                turns += float(error[mask].sum())
                turn_n += int(mask.sum())
                count += len(target)
        record = dict(epoch=epoch, train_loss=float(np.mean(losses)), val_mae=absolute / count,
                      val_weighted_mae=weighted / count,
                      val_turn_mae=turns / max(1, turn_n), val_turn_samples=turn_n,
                      zero_steer_val_mae=zero_error / count)
        history.append(record)
        payload = dict(model_state_dict=model.state_dict(), epoch=epoch, val_mae=record["val_mae"],
                       val_weighted_mae=record["val_weighted_mae"],
                       config=dict(type="camera_history_steering", history_frames=4,
                                   speed_normalization=30.0, image_size=[192, 120]))
        torch.save(payload, args.output / "last.pt")
        if record["val_weighted_mae"] < best:
            best, stale = record["val_weighted_mae"], 0
            torch.save(payload, args.output / "best.pt")
        else:
            stale += 1
        with (args.output / "history.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(record))
            writer.writeheader()
            writer.writerows(history)
        print(f"epoch={epoch} train={record['train_loss']:.4f} MAE={record['val_mae']:.4f} "
              f"weighted={record['val_weighted_mae']:.4f} turn_MAE={record['val_turn_mae']:.4f}", flush=True)
        if stale >= args.patience:
            break
    print(f"Best weighted validation MAE {best:.4f}; saved {args.output / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
