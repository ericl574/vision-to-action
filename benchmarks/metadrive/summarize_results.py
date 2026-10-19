"""Summarize fixed-seed MetaDrive CSVs with the strict safe-success rule.

Older run files stored MetaDrive's raw arrive_dest flag in ``success``;
this reader rechecks collision, offroad, and route completion so those
historical files remain comparable after the benchmark fix.
"""

import argparse
import csv
from pathlib import Path

import numpy as np


def summarize(path: Path) -> dict:
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No episodes in {path}")
    seeds = [int(row["seed"]) for row in rows]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"Duplicate seeds in {path}")
    success_seeds = []
    for row in rows:
        raw_arrival = int(row.get("arrive_dest_raw", row["success"]))
        if (raw_arrival and not int(row["collision"]) and not int(row["offroad"])
                and float(row["route_completion"]) >= 0.95):
            success_seeds.append(int(row["seed"]))
    return dict(
        file=str(path), episodes=len(rows), seed_min=min(seeds), seed_max=max(seeds),
        clean_successes=len(success_seeds), success_rate=len(success_seeds) / len(rows),
        success_seeds=success_seeds,
        collision_rate=float(np.mean([int(row["collision"]) for row in rows])),
        offroad_rate=float(np.mean([int(row["offroad"]) for row in rows])),
        horizon_rate=float(np.mean([int(row["max_steps_reached"]) for row in rows])),
        mean_route_completion=float(np.mean([float(row["route_completion"]) for row in rows])),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.results:
        result = summarize(path)
        print(f"{path}: {result['clean_successes']}/{result['episodes']} "
              f"({result['success_rate']:.1%}) clean successes, "
              f"mean completion {result['mean_route_completion']:.3f}, "
              f"offroad {result['offroad_rate']:.1%}, "
              f"collision {result['collision_rate']:.1%}, "
              f"horizon {result['horizon_rate']:.1%}; "
              f"success seeds {result['success_seeds']}")


if __name__ == "__main__":
    main()
