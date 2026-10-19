# Vision-to-Action

Camera-based imitation learning for closed-loop driving in MetaDrive. This is an independent research and portfolio project, not a production self-driving system or an official leaderboard entry.

[Watch the 43-second driving highlights](benchmarks/metadrive/demo/action_v2_highlights.mp4) · [Full footage of both held-out routes](benchmarks/metadrive/demo/action_v2_two_test_roads.mp4) · [Benchmark implementation](benchmarks/metadrive/run.py)

## What the student actually does

The student receives four RGB frames, spaced 0.5 seconds apart, and ego speed. A CNN-GRU predicts a steering command every 0.1 seconds. A simple proportional speed controller handles acceleration to a fixed 20 km/h target. No LiDAR, lane coordinates, route waypoints, or expert policy outputs enter the student at inference time.

```text
front camera + speed -> CNN-GRU -> steering -> MetaDrive vehicle
                                 speed servo -> acceleration/brake
```

Training uses MetaDrive's privileged PPO expert as a *teacher*. The expert provides action labels during demonstration collection and DAgger-style mixed-policy rollouts. It is not called by the benchmark policy. An earlier comma1M real-camera trajectory model is retained as a domain-shift experiment; its poor closed-loop performance motivated simulator-specific training.

## Measured results

All rows use MetaDrive 0.4.3, 3-block generated roads, traffic density 0.05, fixed seed ranges, and a 20 km/h speed target. A clean success requires arrival, at least 95% route completion, no collision, and no road departure. The stricter definition excludes a MetaDrive edge case that reports arrival at a colliding spawn.

| Split / horizon | Policy | Clean successes | Mean route completion |
| --- | --- | ---: | ---: |
| Development, seeds 5010–5029 / 600 steps | [Zero steering](benchmarks/metadrive/results_zero_development_20.csv) | 0/20 | 0.417 |
| Development, seeds 5010–5029 / 600 steps | [Direct behavior cloning](benchmarks/metadrive/results_action_v1_development_20.csv) | 1/20 | 0.360 |
| Development, seeds 5010–5029 / 600 steps | [Behavior cloning + DAgger](benchmarks/metadrive/results_action_v2_dagger_development_20.csv) | 4/20 | 0.437 |
| First untouched test, seeds 6000–6019 / 600 steps | [Zero steering](benchmarks/metadrive/results_zero_final_test_20.csv) | 0/20 | 0.304 |
| First untouched test, seeds 6000–6019 / 600 steps | [Behavior cloning + DAgger](benchmarks/metadrive/results_action_v2_final_test_20.csv) | 0/20 | 0.412 |
| Separate untouched test, seeds 7000–7019 / 1,000 steps | [Zero steering](benchmarks/metadrive/results_zero_secondary_test_20_h1000.csv) | 2/20 | 0.353 |
| Separate untouched test, seeds 7000–7019 / 1,000 steps | [Behavior cloning + DAgger](benchmarks/metadrive/results_action_v2_secondary_test_20_h1000.csv) | 2/20 | 0.335 |

The two test-route completions shown in the videos are seeds **7000** and **7011**, 224 m and 296 m traveled. Neither of those two roads was completed by zero steering. They are selected successes from the full 20-route test, **not** a 100% success claim. Generalization remains the main open problem. The 1,000-step test is a separately labeled configuration; it was not the original 600-step holdout.

The fixed-seed CSV/JSON files and a strict-success summarizer are under [`benchmarks/metadrive/`](benchmarks/metadrive/). This is a custom reproducible benchmark, not a MetaDrive leaderboard score.

## Reproduce

The project has been run on an Apple Silicon Mac with Python in a `metadrive-ai` Conda environment, PyTorch with MPS, and MetaDrive 0.4.3. Run commands from the repository root. The included checkpoint is only for the compact direct-action model; expert data and larger experiment checkpoints are excluded from Git because of size.

```bash
# Verify coordinate-frame, timing, speed-unit, and success-rule checks.
python -m unittest benchmarks.metadrive.test_pipeline -v

# Run the packaged camera model and a matched zero-steer baseline.
python benchmarks/metadrive/run.py --policy action \
  --checkpoint benchmarks/metadrive/weights/action_v2_dagger_best.pt \
  --seed 7000 --episodes 20 --max-steps 1000 \
  --output benchmarks/metadrive/reproduction_action.csv
python benchmarks/metadrive/run.py --policy zero \
  --seed 7000 --episodes 20 --max-steps 1000 \
  --output benchmarks/metadrive/reproduction_zero.csv
python benchmarks/metadrive/summarize_results.py \
  benchmarks/metadrive/reproduction_action.csv benchmarks/metadrive/reproduction_zero.csv
```

Recording uses the same model input resolution; a larger camera buffer can make the output video clearer without giving the model additional pixels:

```bash
python benchmarks/metadrive/run.py --policy action \
  --checkpoint benchmarks/metadrive/weights/action_v2_dagger_best.pt \
  --seed 7000 --episodes 1 --max-steps 1000 \
  --camera-width 384 --camera-height 240 \
  --record benchmarks/metadrive/demo/reproduced_seed_7000.mp4
```

Training and data-preparation commands, including the expert/DAgger boundary, are documented in [`benchmarks/metadrive/README.md`](benchmarks/metadrive/README.md). The real-data trajectory experiment remains in `prepare_comma1m.py`, `train_comma1m.py`, and `train_comma1m_v2.py`.

## Limitations and next research questions

- A small steering error compounds over time; several roads still end off-road. Open-loop steering accuracy alone did not predict driving success.
- The student has no navigation command, so some intersections or route choices may be ambiguous from camera and speed alone.
- The speed controller is hand-designed. This project does not claim fully learned longitudinal control, real-world transfer, operation in arbitrary simulators, or safety readiness.
- A stronger result would require more diverse corrective data, route-conditioned planning, and a larger untouched benchmark.

## Author

Eric Liu · UBC MEng Mechatronics · BASc Manufacturing Engineering

*Independent project. Not affiliated with Tesla or any autonomous-driving company.*
