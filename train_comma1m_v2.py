
"""V2: CNN-GRU residual trajectory learning on real comma1M recordings."""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

# Reuse the existing, working V1 implementation.
from train_comma1m import DriveSet, CNNGRU


class ResidualCNNGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = CNNGRU()

        # Start exactly at the constant-velocity baseline.
        nn.init.zeros_(self.network.head[-1].weight)
        nn.init.zeros_(self.network.head[-1].bias)

    def forward(self, images, normalized_speed):
        correction = self.network(images, normalized_speed)

        # DriveSet returns speed / 30.
        speed_mps = normalized_speed[:, 0] * 30.0

        times = torch.arange(
            1, 21,
            device=images.device,
            dtype=images.dtype,
        ) * 0.25

        baseline = torch.zeros(
            (images.shape[0], 20, 2),
            dtype=images.dtype,
            device=images.device,
        )
        baseline[:, :, 0] = speed_mps[:, None] * times[None, :]

        return baseline + correction, baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="data/processed/comma1m_v1/samples.csv",
    )
    parser.add_argument(
        "--v1-split",
        default="checkpoints/comma1m_cnn_gru_v1/split.json",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--output",
        default="checkpoints/comma1m_cnn_gru_v2",
    )
    args = parser.parse_args()

    random.seed(27)
    np.random.seed(27)
    torch.manual_seed(27)

    with open(args.data, newline="") as f:
        rows = list(csv.DictReader(f))

    with open(args.v1_split) as f:
        split = json.load(f)

    train_ids = set(split["train"])
    validation_ids = set(split["validation"])

    if train_ids & validation_ids:
        raise RuntimeError("Training and validation segments overlap.")

    available_ids = {row["segment"] for row in rows}

    if available_ids != train_ids | validation_ids:
        raise RuntimeError(
            "Dataset segments differ from the original V1 split."
        )

    train_rows = [
        row for row in rows
        if row["segment"] in train_ids
    ]
    val_rows = [
        row for row in rows
        if row["segment"] in validation_ids
    ]

    if not train_rows or not val_rows:
        raise RuntimeError("Empty training or validation dataset.")

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    train_loader = DataLoader(
        DriveSet(train_rows),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        DriveSet(val_rows),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)

    (output / "split.json").write_text(
        json.dumps(split, indent=2)
    )

    model = ResidualCNNGRU().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    best_ade = float("inf")
    stale_epochs = 0
    history = []

    print(
        f"Train {len(train_rows)} samples / {len(train_ids)} segments"
    )
    print(
        f"Validation {len(val_rows)} samples / "
        f"{len(validation_ids)} segments"
    )
    print("Device:", device)
    print("Residual initialization: constant velocity")
    print("V1 split reused:", args.v1_split, flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0

        for images, speed, target in train_loader:
            images = images.to(device)
            speed = speed.to(device)
            target = target.to(device)

            prediction, _ = model(images, speed)

            loss = nn.functional.smooth_l1_loss(
                prediction, target, beta=2.0
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item() * len(images)

        train_loss = train_loss_sum / len(train_rows)

        model.eval()

        total_ade = 0.0
        total_fde = 0.0
        total_base_ade = 0.0
        total_base_fde = 0.0
        total_samples = 0

        with torch.inference_mode():
            for images, speed, target in val_loader:
                images = images.to(device)
                speed = speed.to(device)
                target = target.to(device)

                prediction, baseline = model(images, speed)

                pred_dist = torch.linalg.vector_norm(
                    prediction - target, dim=-1
                )
                base_dist = torch.linalg.vector_norm(
                    baseline - target, dim=-1
                )

                total_ade += pred_dist.sum().item()
                total_fde += pred_dist[:, -1].sum().item()

                total_base_ade += base_dist.sum().item()
                total_base_fde += base_dist[:, -1].sum().item()

                total_samples += len(images)

        val_ade = total_ade / (total_samples * 20)
        val_fde = total_fde / total_samples
        base_ade = total_base_ade / (total_samples * 20)
        base_fde = total_base_fde / total_samples

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "ADE_m": val_ade,
            "FDE_m": val_fde,
            "baseline_ADE_m": base_ade,
            "baseline_FDE_m": base_fde,
        }
        history.append(record)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "val_ade": val_ade,
            "config": {
                "type": "residual_cnn_gru_trajectory",
                "history_frames": 4,
                "future_waypoints": 20,
                "speed_normalization": 30.0,
            },
        }

        torch.save(checkpoint, output / "last.pt")

        if val_ade < best_ade:
            best_ade = val_ade
            stale_epochs = 0
            torch.save(checkpoint, output / "best.pt")
        else:
            stale_epochs += 1

        print(
            f"Epoch {epoch:02d} "
            f"train={train_loss:.4f} "
            f"ADE={val_ade:.2f}m "
            f"FDE={val_fde:.2f}m | "
            f"baseline ADE={base_ade:.2f}m "
            f"FDE={base_fde:.2f}m",
            flush=True,
        )

        if stale_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    with open(output / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=history[0].keys()
        )
        writer.writeheader()
        writer.writerows(history)

    print(f"Best validation ADE: {best_ade:.2f}m")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
