"""Turn collected route directories into training samples. Split by route, never by frame."""

import json
import random
from pathlib import Path

import numpy as np

from benchmarks.carla.record import load_manifest, read_labels

CLEAN_STATUSES = ("arrived", "timeout")
TAIL_DROP_FRAMES = 10  # the last second before a crash/off-road is not expert behaviour worth imitating


def history_indices(index: int, history: int, stride: int) -> list[int]:
    return [max(0, index - k * stride) for k in reversed(range(history))]


def route_samples(route_dir, history: int, stride: int, frame_step: int = 1) -> list[dict]:
    route_dir = Path(route_dir)
    with open(route_dir / "meta.json") as handle:
        meta = json.load(handle)
    labels = read_labels(route_dir)
    end = len(labels) if meta.get("status") in CLEAN_STATUSES else max(0, len(labels) - TAIL_DROP_FRAMES)
    samples = []
    for i in range(0, end, frame_step):
        row = labels[i]
        frames = [str(route_dir / "frames" / f"{j:06d}.jpg") for j in history_indices(i, history, stride)]
        samples.append({
            "frames": frames,
            "speed": float(row["speed_mps"]),
            "command": int(row["command"]),
            "steer": float(row["steer"]),
            "target_speed": float(row["target_speed_mps"]),
            "tl_state": int(row["tl_state"]),
            "hazard": int(row["hazard"]),
            "hazard_dist": float(row["hazard_dist_m"]) if row.get("hazard_dist_m") not in ("", None) else 0.0,
            "route_id": meta["route_id"],
        })
    return samples


def build_samples(roots, history: int = 3, stride: int = 3, val_frac: float = 0.15, split_file=None,
                  seed: int = 0, frame_step: int = 1):
    routes = []
    for root in roots:
        routes.extend(load_manifest(root))
    if not routes:
        raise RuntimeError(f"No finished routes under {roots}")
    route_ids = sorted({route["route_id"] for route in routes})
    if split_file and Path(split_file).exists():
        with open(split_file) as handle:
            split = json.load(handle)
        val_ids = set(split["val"])
        # Routes unseen by the saved split (e.g. new DAgger data) go to training.
        train_ids = set(route_ids) - val_ids
    else:
        rng = random.Random(seed)
        shuffled = list(route_ids)
        rng.shuffle(shuffled)
        n_val = max(1, round(len(shuffled) * val_frac))
        val_ids = set(shuffled[:n_val])
        train_ids = set(shuffled[n_val:])
    train_rows, val_rows = [], []
    for route in routes:
        samples = route_samples(route["dir"], history, stride, frame_step)
        (val_rows if route["route_id"] in val_ids else train_rows).extend(samples)
    split = {"train": sorted(train_ids), "val": sorted(val_ids)}
    return train_rows, val_rows, split


def sample_weights(rows: list[dict]) -> np.ndarray:
    """Oversample the rare, decision-heavy frames: braking, turning, red lights."""
    weights = np.ones(len(rows), dtype=np.float64)
    for i, row in enumerate(rows):
        if row["target_speed"] < 1.0:
            weights[i] *= 4.0
        if abs(row["steer"]) > 0.1 or row["command"] in (1, 2):
            weights[i] *= 3.0
        if row["tl_state"] == 3:
            weights[i] *= 3.0
    return weights


def create_dataset(rows: list[dict], width: int, height: int, augment: bool):
    import cv2
    import torch
    from torch.utils.data import Dataset

    class DriveSet(Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            row = rows[index]
            frames = []
            gain = np.random.uniform(0.8, 1.2) if augment else 1.0
            bias = np.random.uniform(-15, 15) if augment else 0.0
            for path in row["frames"]:
                bgr = cv2.imread(path, cv2.IMREAD_COLOR)
                if bgr is None:
                    raise FileNotFoundError(path)
                if bgr.shape[0] != height or bgr.shape[1] != width:
                    bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
                rgb = bgr[:, :, ::-1].astype(np.float32)
                if augment:
                    rgb = np.clip(rgb * gain + bias, 0, 255)
                frames.append(rgb / 255.0)
            images = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()
            return {
                "images": images,
                "speed": torch.tensor([row["speed"]], dtype=torch.float32),
                "command": torch.tensor(row["command"], dtype=torch.long),
                "steer": torch.tensor(row["steer"], dtype=torch.float32),
                "target_speed": torch.tensor(row["target_speed"], dtype=torch.float32),
                "tl_state": torch.tensor(row["tl_state"], dtype=torch.long),
                "hazard": torch.tensor(float(row["hazard"]), dtype=torch.float32),
                "hazard_dist": torch.tensor(row["hazard_dist"], dtype=torch.float32),
            }

    return DriveSet()


def describe(rows: list[dict]) -> dict:
    n = max(1, len(rows))
    commands = np.bincount([row["command"] for row in rows], minlength=4).tolist() if rows else [0, 0, 0, 0]
    return {
        "samples": len(rows),
        "routes": len({row["route_id"] for row in rows}),
        "braking_frac": sum(1 for r in rows if r["target_speed"] < 1.0) / n,
        "red_light_frac": sum(1 for r in rows if r["tl_state"] == 3) / n,
        "hazard_frac": sum(1 for r in rows if r["hazard"]) / n,
        "turn_frac": sum(1 for r in rows if abs(r["steer"]) > 0.1) / n,
        "commands": dict(zip(("follow", "left", "right", "straight"), commands)),
    }
