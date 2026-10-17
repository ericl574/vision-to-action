"""Load Udacity simulator camera frames and steering labels."""
import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DrivingDataset(Dataset):
    def __init__(self, data_dir: str | Path, records: list, image_size=(96, 192)):
        self.data_dir = Path(data_dir)
        self.records = records
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),  # converts RGB pixels to float32 in [0, 1]
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        img_path, steering = self.records[index]
        with Image.open(img_path) as image:
            x = self.transform(image.convert("RGB"))
        y = torch.tensor([steering], dtype=torch.float32)
        return x, y


def read_records(data_dir: str | Path):
    """Read center-camera rows from data/driving_log.csv.

    Works with a CSV with or without a header, and with absolute, relative,
    or Windows-style IMG paths. Deliberately ignores side-camera recordings
    for the first baseline: no unverified steering-angle offsets.
    """
    data_dir = Path(data_dir)
    log_path = data_dir / "driving_log.csv"
    if not log_path.is_file():
        raise FileNotFoundError(
            f"Missing {log_path}. Expected data/driving_log.csv and data/IMG/."
        )

    records = []
    missing = 0
    with log_path.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.reader(file):
            if len(row) < 4:
                continue
            raw_path = row[0].strip().strip('"').replace('\\', '/')
            try:
                steering = float(row[3])
            except ValueError:
                continue  # Header row

            filename = Path(raw_path).name
            image_path = data_dir / "IMG" / filename
            if not image_path.is_file():
                alternative = data_dir / raw_path
                image_path = alternative if alternative.is_file() else image_path

            if image_path.is_file():
                records.append((image_path, steering))
            else:
                missing += 1

    if len(records) < 20:
        raise RuntimeError(
            f"Only {len(records)} usable images found ({missing} missing). "
            f"Check that {data_dir}/IMG/ contains image files."
        )
    print(f"Dataset: {len(records)} center-camera frames ({missing} missing).")
    return records


def chronological_split(records, val_fraction=0.2):
    """Hold out the last 20% of the recording, avoiding adjacent-frame leakage."""
    split_index = max(1, int(len(records) * (1 - val_fraction)))
    return records[:split_index], records[split_index:]
