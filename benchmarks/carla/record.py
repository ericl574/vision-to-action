"""On-disk layout for collected routes: ``<root>/<route_id>/frames/000123.jpg`` + ``labels.csv`` + ``meta.json``.

A route directory without ``meta.json`` is an interrupted collection and is ignored by
``load_manifest`` so a killed collector never corrupts a dataset.
"""

import csv
import json
from pathlib import Path

LABEL_COLUMNS = [
    "idx", "t", "speed_mps", "command", "steer", "throttle", "brake", "target_speed_mps", "emergency",
    "tl_state", "tl_dist_m", "hazard", "hazard_dist_m", "hazard_type", "is_perturbed",
    "exec_steer", "exec_throttle", "exec_brake", "used_expert",
    "student_steer", "student_target_speed_mps", "route_completion", "x", "y", "yaw",
]


# Per-step network outputs logged by run.py for the overlay (no privileged fields except the gt_* columns).
OUTPUT_COLUMNS = [
    "idx", "t", "speed_mps", "command", "steer", "throttle", "brake", "target_speed_mps",
    "tl_none", "tl_green", "tl_yellow", "tl_red", "hazard_prob", "hazard_dist_m",
    "gt_tl_state", "gt_hazard", "route_completion", "ran_red_light", "x", "y",
]


class RouteRecorder:
    def __init__(self, root, route_id: str, meta: dict | None = None, jpeg_quality: int = 90,
                 columns: list[str] | None = None, csv_name: str = "labels.csv"):
        self.dir = Path(root) / route_id
        self.frames_dir = self.dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.meta = dict(meta or {})
        self.meta["route_id"] = route_id
        self.rows: list[dict] = []
        self.jpeg_quality = jpeg_quality
        self.columns = list(columns or LABEL_COLUMNS)
        self.csv_name = csv_name
        if (self.dir / "meta.json").exists():
            (self.dir / "meta.json").unlink()

    def append(self, rgb, row: dict) -> None:
        import cv2

        index = len(self.rows)
        path = self.frames_dir / f"{index:06d}.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        record = {"idx": index}
        record.update(row)
        self.rows.append(record)

    def finish(self, status: str, extra: dict | None = None) -> Path:
        with open(self.dir / self.csv_name, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.columns, extrasaction="ignore")
            writer.writeheader()
            for row in self.rows:
                writer.writerow({key: row.get(key, "") for key in self.columns})
        self.meta.update({"status": status, "n_frames": len(self.rows)})
        if extra:
            self.meta.update(extra)
        with open(self.dir / "meta.json", "w") as handle:
            json.dump(self.meta, handle, indent=1)
        return self.dir


def load_manifest(root) -> list[dict]:
    """Return metadata for every finished route under ``root`` (recursively), sorted by path."""
    root = Path(root)
    entries = []
    for meta_path in sorted(root.rglob("meta.json")):
        with open(meta_path) as handle:
            meta = json.load(handle)
        meta["dir"] = str(meta_path.parent)
        entries.append(meta)
    return entries


def read_labels(route_dir) -> list[dict]:
    with open(Path(route_dir) / "labels.csv", newline="") as handle:
        return list(csv.DictReader(handle))
