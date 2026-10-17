"""Evaluate the best checkpoint on the held-out chronological data."""
import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import DrivingDataset, chronological_split, read_records
from model import DrivingModel
from train import select_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=0, help="Match --limit used in train.py")
    args = parser.parse_args()
    records = read_records(args.data)
    if args.limit > 0:
        records = records[:args.limit]
    train_rows, val_rows = chronological_split(records)
    loader = DataLoader(DrivingDataset(args.data, val_rows), batch_size=64,
                        shuffle=False, num_workers=0)
    device = select_device(args.device)
    ckpt_path = Path("checkpoints/best.pt")
    if not ckpt_path.is_file():
        raise FileNotFoundError("Missing checkpoints/best.pt. Run python train.py first.")
    model = DrivingModel().to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    preds, labels = [], []
    with torch.no_grad():
        for images, targets in loader:
            predictions = model(images.to(device)).cpu().flatten()
            preds.extend(predictions.tolist())
            labels.extend(targets.flatten().tolist())

    rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(preds, labels)) / len(labels))
    mae = sum(abs(a - b) for a, b in zip(preds, labels)) / len(labels)
    baseline = math.sqrt(sum(y ** 2 for y in labels) / len(labels))
    train_mean = sum(y for _, y in train_rows) / len(train_rows)
    baseline_mean = math.sqrt(sum((y-train_mean)**2 for y in labels) / len(labels))
    print(f"Checkpoint epoch: {checkpoint['epoch']}, held-out frames: {len(labels)}")
    print(f"Model RMSE: {rmse:.5f}; MAE: {mae:.5f}")
    print(f"Zero-steering RMSE: {baseline:.5f}")
    print(f"Train-mean predictor RMSE: {baseline_mean:.5f}")
    print("First 10: prediction → reference steering")
    for a, b in list(zip(preds, labels))[:10]:
        print(f"  {a:+.4f} → {b:+.4f}")
    print("CAUTION: These numbers are offline metrics, not closed-loop driving safety.")


if __name__ == "__main__":
    main()
