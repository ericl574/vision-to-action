"""Train the user's DrivingModel on Udacity simulator demonstration frames."""
import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import DrivingDataset, chronological_split, read_records
from model import DrivingModel


def select_device(choice):
    if choice != "auto":
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate_epoch(model, dataloader, device):
    model.eval()
    total_sq, total_abs, count = 0.0, 0.0, 0
    with torch.no_grad():
        for images, target in dataloader:
            images, target = images.to(device), target.to(device)
            error = model(images) - target
            total_sq += error.square().sum().item()
            total_abs += error.abs().sum().item()
            count += target.numel()
    return math.sqrt(total_sq / count), total_abs / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=0, help="Optional quick-test frame limit")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    records = read_records(args.data)
    if args.limit > 0:
        records = records[:args.limit]
        if len(records) < 20:
            raise ValueError("Need at least 20 images for a training/validation split")
    train_rows, val_rows = chronological_split(records)
    print(f"Chronological split: train={len(train_rows)}, validation={len(val_rows)}")
    print("Important: This is an OFFLINE steering prediction baseline, not a driving demo.")

    train_loader = DataLoader(DrivingDataset(args.data, train_rows),
                              batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(DrivingDataset(args.data, val_rows),
                            batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = select_device(args.device)
    model = DrivingModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    output = Path("experiments")
    checkpoint = Path("checkpoints")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint.mkdir(parents=True, exist_ok=True)

    baseline_zero_rmse = math.sqrt(sum(y * y for _, y in val_rows) / len(val_rows))
    mean_steering = sum(y for _, y in train_rows) / len(train_rows)
    baseline_mean_rmse = math.sqrt(
        sum((y - mean_steering) ** 2 for _, y in val_rows) / len(val_rows)
    )
    print(f"Device: {device}; zero-steer RMSE={baseline_zero_rmse:.5f}; "
          f"train-mean RMSE={baseline_mean_rmse:.5f}")

    history_path = output / "history.csv"
    best_rmse = float("inf")
    with history_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_rmse", "val_rmse", "val_mae"])
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_sse, train_count = 0.0, 0
            for images, target in train_loader:
                images, target = images.to(device), target.to(device)
                optimizer.zero_grad(set_to_none=True)
                predictions = model(images)
                loss = criterion(predictions, target)
                loss.backward()
                optimizer.step()
                train_sse += loss.item() * target.numel()
                train_count += target.numel()

            train_rmse = math.sqrt(train_sse / train_count)
            val_rmse, val_mae = evaluate_epoch(model, val_loader, device)
            writer.writerow([epoch, train_rmse, val_rmse, val_mae])
            file.flush()
            print(f"Epoch {epoch}/{args.epochs} | train RMSE {train_rmse:.5f} | "
                  f"val RMSE {val_rmse:.5f} | val MAE {val_mae:.5f}")
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                torch.save({"model_state_dict": model.state_dict(),
                            "epoch": epoch, "val_rmse": val_rmse}, checkpoint / "best.pt")

    print(f"Saved best checkpoint: {checkpoint / 'best.pt'}")
    print(f"Saved learning curve: {history_path}")
    print(f"Best validation RMSE: {best_rmse:.5f} "
          f"(zero-steer: {baseline_zero_rmse:.5f}; train-mean: {baseline_mean_rmse:.5f})")
    print("Next: python evaluate.py")


if __name__ == "__main__":
    main()
