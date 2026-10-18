
"""Train and resume the Vision-to-Action driving model."""

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

    with torch.inference_mode():
        for images, target in dataloader:
            images = images.to(device)
            target = target.to(device)

            error = model(images) - target
            total_sq += error.square().sum().item()
            total_abs += error.abs().sum().item()
            count += target.numel()

    return math.sqrt(total_sq / count), total_abs / count


def save_checkpoint(path, model, optimizer, epoch,
                    val_rmse, best_rmse, history_path,
                    training_config):

    # Convert NumPy RNG state into safe Python values.
    np_state = np.random.get_state()

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_rmse": val_rmse,
        "best_val_rmse": best_rmse,
        "history_path": str(history_path),
        "training_config": training_config,
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": (
            np_state[0],
            np_state[1].tolist(),
            np_state[2],
            np_state[3],
            np_state[4],
        ),
    }

    if torch.backends.mps.is_available():
        state["mps_rng_state"] = torch.mps.get_rng_state()

    # Atomic replacement protects previous checkpoints.
    temp_path = path.with_name(path.name + ".tmp")
    torch.save(state, temp_path)
    temp_path.replace(path)


def restore_rng(state):
    if "torch_rng_state" in state:
        torch.set_rng_state(state["torch_rng_state"])

    if "python_rng_state" in state:
        random.setstate(state["python_rng_state"])

    if "numpy_rng_state" in state:
        name, keys, pos, has_gauss, cached = (
            state["numpy_rng_state"]
        )
        np.random.set_state((
            name,
            np.array(keys, dtype=np.uint32),
            pos,
            has_gauss,
            cached,
        ))

    if (
        "mps_rng_state" in state
        and torch.backends.mps.is_available()
    ):
        torch.mps.set_rng_state(state["mps_rng_state"])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default="data")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint; epochs means ADDITIONAL epochs",
    )

    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    if args.lr <= 0 or args.limit < 0:
        parser.error("lr must be positive and limit nonnegative")

    # Reproducible initial state
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # -------- Dataset --------

    records = read_records(args.data)

    if args.limit > 0:
        records = records[:args.limit]
        if len(records) < 20:
            raise ValueError("Need at least 20 images")

    train_rows, val_rows = chronological_split(records)

    print(
        f"Chronological split: train={len(train_rows)}, "
        f"validation={len(val_rows)}"
    )
    print("Offline steering prediction + simulator control.")

    train_loader = DataLoader(
        DrivingDataset(args.data, train_rows),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        DrivingDataset(args.data, val_rows),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # -------- Model --------

    device = select_device(args.device)
    model = DrivingModel().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    criterion = nn.MSELoss()

    output_dir = Path("experiments")
    checkpoint_dir = Path("checkpoints")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"

    start_epoch = 0
    best_rmse = float("inf")
    history_path = output_dir / "history.csv"
    history_mode = "w"

    training_config = {
        "data": str(Path(args.data).resolve()),
        "limit": args.limit,
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "batch_size": args.batch_size,
    }

    # -------- Resume Training --------

    if args.resume:
        state = torch.load(
            args.resume,
            map_location="cpu",
            weights_only=True,
        )

        model.load_state_dict(state["model_state_dict"])

        start_epoch = int(state["epoch"])
        best_rmse = float(
            state.get("best_val_rmse", state["val_rmse"])
        )

        if "training_config" in state:
            saved_config = state["training_config"]

            if saved_config != training_config:
                raise ValueError(
                    "Dataset or batch configuration differs "
                    "from the saved checkpoint."
                )

        if "optimizer_state_dict" in state:
            optimizer.load_state_dict(
                state["optimizer_state_dict"]
            )

            restore_rng(state)

            print("Model weights restored!")
            print("Optimizer state restored!")
            print("Random states restored where available.")
            print(
                "Restored learning rate:",
                optimizer.param_groups[0]["lr"],
            )

            history_path = Path(
                state.get(
                    "history_path",
                    output_dir / f"resume_from_epoch_{start_epoch}.csv",
                )
            )

            if history_path.exists():
                with history_path.open(newline="") as f:
                    existing_rows = list(csv.DictReader(f))

                if existing_rows:
                    last_logged = int(
                        existing_rows[-1]["epoch"]
                    )
                    if last_logged != start_epoch:
                        raise ValueError(
                            f"History ends at epoch {last_logged}, "
                            f"but checkpoint is epoch {start_epoch}. "
                            "Reconcile the history before resuming."
                        )

                history_mode = "a"

        else:
            # Legacy best.pt contains model weights only.
            history_path = (
                output_dir / f"resume_from_epoch_{start_epoch}.csv"
            )

            if history_path.exists():
                raise FileExistsError(
                    f"{history_path} already exists. "
                    "Archive it before starting a new legacy resume."
                )

            print("Legacy checkpoint loaded!")
            print("Model weights restored.")
            print("Optimizer state starts fresh.")

        print(f"Resuming from Epoch {start_epoch}")
        print(f"Previous best RMSE: {best_rmse:.5f}")

    else:
        print("Starting new training run.")

        if best_path.exists() or last_path.exists():
            raise FileExistsError(
                "Existing checkpoints detected. "
                "Use --resume or archive the checkpoints "
                "before starting a fresh run."
            )

    # -------- Baselines --------

    baseline_zero_rmse = math.sqrt(
        sum(y * y for _, y in val_rows) / len(val_rows)
    )

    mean_steering = (
        sum(y for _, y in train_rows) / len(train_rows)
    )

    baseline_mean_rmse = math.sqrt(
        sum(
            (y - mean_steering) ** 2
            for _, y in val_rows
        ) / len(val_rows)
    )

    print(f"Device: {device}")
    print(f"Zero-steer RMSE: {baseline_zero_rmse:.5f}")
    print(f"Train-mean RMSE: {baseline_mean_rmse:.5f}")

    # -------- Training Loop --------

    end_epoch = start_epoch + args.epochs

    print(f"Training Epoch {start_epoch + 1} to {end_epoch}")

    with history_path.open(history_mode, newline="") as file:
        writer = csv.writer(file)

        if history_mode == "w":
            writer.writerow([
                "epoch",
                "train_rmse",
                "val_rmse",
                "val_mae",
            ])
            file.flush()

        for epoch in range(start_epoch + 1, end_epoch + 1):

            model.train()
            train_sse = 0.0
            train_count = 0

            for images, target in train_loader:
                images = images.to(device)
                target = target.to(device)

                optimizer.zero_grad(set_to_none=True)

                predictions = model(images)
                loss = criterion(predictions, target)

                loss.backward()
                optimizer.step()

                train_sse += loss.item() * target.numel()
                train_count += target.numel()

            train_rmse = math.sqrt(
                train_sse / train_count
            )

            val_rmse, val_mae = evaluate_epoch(
                model, val_loader, device
            )

            print(
                f"Epoch {epoch}/{end_epoch} | "
                f"train RMSE {train_rmse:.5f} | "
                f"val RMSE {val_rmse:.5f} | "
                f"val MAE {val_mae:.5f}"
            )

            # Track the best validation model.
            improved = val_rmse < best_rmse

            if improved:
                best_rmse = val_rmse

            # Save last.pt EVERY epoch.
            save_checkpoint(
                last_path,
                model,
                optimizer,
                epoch,
                val_rmse,
                best_rmse,
                history_path,
                training_config,
            )

            # Save best.pt ONLY when validation improves.
            if improved:
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    epoch,
                    val_rmse,
                    best_rmse,
                    history_path,
                    training_config,
                )
                print("  New best model saved!")

            writer.writerow([
                epoch,
                train_rmse,
                val_rmse,
                val_mae,
            ])
            file.flush()

    # -------- Summary --------

    print("\nTraining completed!")
    print(f"Final epoch: {end_epoch}")
    print(f"Best validation RMSE: {best_rmse:.5f}")
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")
    print(f"Training history: {history_path}")
    print("Next: python evaluate.py")


if __name__ == "__main__":
    main()
