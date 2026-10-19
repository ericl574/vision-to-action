"""Index successful MetaDrive expert images and steering labels for imitation learning."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/collected/metadrive_expert_v1"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/metadrive_actions_v1"))
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()
    if args.stride < 1 or args.output.exists():
        parser.error("stride must be positive and output directory must not exist")
    samples, seeds = [], []
    for summary_path in sorted(args.source.glob("seed_*/episodes.csv")):
        with summary_path.open(newline="") as file:
            summaries = list(csv.DictReader(file))
        for summary in summaries:
            seed = int(summary["seed"])
            if seed >= 5000:
                raise ValueError(f"Evaluation seed in training source: {seed}")
            corrective_route = summary_path.parent.name.endswith(("_recovery", "_dagger"))
            if int(summary["success"]) != 1 and not corrective_route:
                continue
            episode = summary_path.parent / f"episode_{seed}"
            with (episode / "actions.csv").open(newline="") as file:
                actions = list(csv.DictReader(file))
            if corrective_route and int(summary["offroad"]) and len(actions) > 10:
                actions = actions[:-5]  # Avoid the final, potentially unrecoverable boundary states.
            if not actions:
                continue
            seeds.append(seed)
            for index in range(0, len(actions), args.stride):
                row = actions[index]
                speed_key = "speed_mps" if "speed_mps" in row else "speed_kmh"
                # Legacy speed_kmh is mislabeled: agent.speed is always m/s.
                history = [str(episode / actions[max(index - 5 * n, 0)]["image"])
                           for n in (3, 2, 1, 0)]
                samples.append(dict(segment=episode.name, index=index,
                                    history=json.dumps(history), speed=row[speed_key],
                                    steering=row["steering"]))
    if len(seeds) < 5:
        raise RuntimeError("Need at least five expert routes")
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "samples.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)
    (args.output / "summary.json").write_text(json.dumps(dict(
        samples=len(samples), routes=len(seeds), seeds=seeds,
        history_spacing_s=0.5, stride_steps=args.stride,
        student_inputs=["four RGB frames", "ego speed m/s"],
        target="expert steering at current pre-step image",
    ), indent=2))
    print(f"Prepared {len(samples)} action samples from {len(seeds)} expert routes", flush=True)


if __name__ == "__main__":
    main()
