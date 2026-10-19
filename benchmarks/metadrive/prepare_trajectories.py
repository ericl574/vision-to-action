"""Turn MetaDrive expert pose logs into camera-to-5-second trajectory samples.

Each input image is captured before its corresponding expert action. The
targets are positions at +0.25, ..., +5.0 seconds in the current vehicle
heading frame: forward and right in metres. No privileged fields become
student inputs. Only complete, successful teacher routes are used.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def make_samples(episode_dir: Path, stride: int) -> list[dict]:
    with (episode_dir / "actions.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) < 51 or "position_x" not in rows[0]:
        return []
    positions = np.asarray([[float(row["position_x"]), float(row["position_y"])] for row in rows])
    times = np.arange(1, 21, dtype=np.float64) * 2.5  # 0.25 s / 0.1 s
    samples = []
    for index in range(0, len(rows) - 50, stride):
        # Earlier collector runs mislabeled MetaDrive's m/s value as km/h.
        speed_key = "speed_mps" if "speed_mps" in rows[index] else "speed_kmh"
        speed = float(rows[index][speed_key])
        heading = float(rows[index]["heading_theta"])
        forward = np.array([np.cos(heading), np.sin(heading)])
        right = np.array([forward[1], -forward[0]])
        future = np.stack([
            np.interp(index + times, np.arange(len(rows)), positions[:, axis])
            for axis in (0, 1)
        ], axis=-1)
        displacement = future - positions[index]
        trajectory = np.stack([displacement @ forward, displacement @ right], axis=-1)
        history = [str(episode_dir / rows[max(index - 5 * n, 0)]["image"])
                   for n in (3, 2, 1, 0)]
        samples.append(dict(
            segment=episode_dir.name,
            index=index,
            history=json.dumps(history),
            speed=f"{speed:.6f}",
            trajectory=json.dumps(trajectory.round(5).tolist()),
        ))
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/collected/metadrive_expert_v1"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/metadrive_trajectories_v1"))
    parser.add_argument("--stride", type=int, default=5)
    args = parser.parse_args()
    if args.stride < 1 or args.output.exists():
        parser.error("stride must be positive and output must not exist")
    samples, included, skipped = [], [], []
    for action_path in sorted(args.source.glob("seed_*/episode_*/actions.csv")):
        if action_path.parent.parent.name.endswith(("_recovery", "_dagger")):
            continue  # Future path would include injected disturbances.
        episode_dir = action_path.parent
        seed = int(episode_dir.name.removeprefix("episode_"))
        if seed >= 5000:
            raise ValueError(f"Evaluation seed found in training source: {seed}")
        summary_path = episode_dir.parent / "episodes.csv"
        with summary_path.open(newline="") as file:
            summaries = {int(row["seed"]): row for row in csv.DictReader(file)}
        if seed not in summaries or int(summaries[seed]["success"]) != 1:
            skipped.append(seed)
            continue
        episode_samples = make_samples(episode_dir, args.stride)
        if episode_samples:
            samples.extend(episode_samples)
            included.append(seed)
        else:
            skipped.append(seed)
    if not samples:
        raise RuntimeError("No complete successful pose-labeled episodes found")
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "samples.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)
    (args.output / "summary.json").write_text(json.dumps(dict(
        samples=len(samples), included_seeds=included, skipped_seeds=skipped,
        stride_steps=args.stride, history_spacing_s=0.5, future_spacing_s=0.25,
        coordinate_frame="current vehicle heading; x forward, y right",
        student_inputs=["four RGB frames", "ego speed"],
    ), indent=2))
    print(f"Prepared {len(samples)} samples from {len(included)} successful expert episodes; "
          f"skipped {len(skipped)}", flush=True)


if __name__ == "__main__":
    main()
