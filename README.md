# Vision-to-Action

Camera-based imitation learning for closed-loop driving, first in MetaDrive and now in CARLA. This is an independent research and portfolio project, not a production self-driving system or an official leaderboard entry.

## CARLA: end-to-end camera driving with traffic lights, turns and pedestrians

A network trained here drives a car through CARLA's Town10HD from **front-camera frames, its own speed and a sat-nav-style turn command only**. It outputs steering and desired speed directly; no hand-written traffic-light or pedestrian rules exist in the control path. Teacher: CARLA's privileged `BehaviorAgent`. Training: behaviour cloning, then DAgger.

On a 15-route held-out test under a strict rule (arrive, no collision, no off-road, **no red-light violation**), the best camera-only model (pilot v2) completed **10/15** routes cleanly with **zero red-light violations**. The privileged expert scored 11/15 under the same rule. Behaviour cloning alone scored 5/15.

[Watch the held-out demo reel](benchmarks/carla/demo/pilot_v2_heldout_reel.mp4) · [Full method, results and limitations](benchmarks/carla/README.md)

## MetaDrive (earlier experiment)

[Watch a v3 held-out route (seed 8013)](benchmarks/metadrive/demo/action_v3_final_test_seed_8013.mp4) · [Second v3 route (seed 8018)](benchmarks/metadrive/demo/action_v3_final_test_seed_8018.mp4) · [Earlier v2 highlights](benchmarks/metadrive/demo/action_v2_highlights.mp4)

## What the Learner model actually does

The Learning Model receives four RGB frames, spaced 0.5 seconds apart, and ego speed. A CNN-GRU predicts a steering command every 0.1 seconds. A simple proportional speed controller handles acceleration to a fixed 20 km/h target. No LiDAR, lane coordinates, route waypoints, or expert policy outputs enter the student at inference time.

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
| Development, seeds 5010–5029 / 1,000 steps | [v2 DAgger](benchmarks/metadrive/results_action_v2_dagger_development_20_h1000.csv) | 4/20 | 0.437 |
| Development, seeds 5010–5029 / 1,000 steps | [v3 DAgger](benchmarks/metadrive/results_action_v3_dagger_development_20_h1000.csv) | 7/20 | 0.615 |
| First untouched test, seeds 6000–6019 / 600 steps | [Zero steering](benchmarks/metadrive/results_zero_final_test_20.csv) | 0/20 | 0.304 |
| First untouched test, seeds 6000–6019 / 600 steps | [Behavior cloning + DAgger](benchmarks/metadrive/results_action_v2_final_test_20.csv) | 0/20 | 0.412 |
| Separate untouched test, seeds 7000–7019 / 1,000 steps | [Zero steering](benchmarks/metadrive/results_zero_secondary_test_20_h1000.csv) | 2/20 | 0.353 |
| Separate untouched test, seeds 7000–7019 / 1,000 steps | [Behavior cloning + DAgger](benchmarks/metadrive/results_action_v2_secondary_test_20_h1000.csv) | 2/20 | 0.335 |
| Final held-out test, seeds 8000–8019 / 1,000 steps | [Zero steering](benchmarks/metadrive/results_zero_final_test_20_h1000.csv) | 1/20 | 0.295 |
| Final held-out test, seeds 8000–8019 / 1,000 steps | [Selected v3 DAgger model](benchmarks/metadrive/results_action_v3_final_test_20_h1000.csv) | 3/20 | 0.545 |

The selected checkpoint is [`action_v3_dagger_best.pt`](benchmarks/metadrive/weights/action_v3_dagger_best.pt); the earlier v2 checkpoint and videos remain available. On the matched development set, v3 improved clean completion (7/20 vs. v2's 4/20), mean route completion (0.615 vs. 0.437), and road-departure rate (60% vs. 80%), but had 2 collisions versus none. On fresh seeds 8000–8019, v3 achieved 3/20 clean completions versus 1/20 for zero steering, with mean route completion 0.545 vs. 0.295, road departures 80% vs. 95%, and collisions 10% for both. Neither policy timed out in that test. These results show a measurable but modest improvement, not robust driving.

The two v3 videos above are successful routes selected from the full 20-seed final test: seed **8013** completed 98.8% of its route over 392.5 m; seed **8018** completed 99.2% over 530.3 m. They illustrate successful cases and should be read alongside the full test table, not as a typical-run or 100% success claim. The earlier v2 videos are retained as historical artifacts.

The fixed-seed CSV/JSON files and a strict-success summarizer are under [`benchmarks/metadrive/`](benchmarks/metadrive/). This is a custom reproducible benchmark, not a MetaDrive leaderboard score.

## Reproduce

The project has been run on an Apple Silicon Mac with Python 3.11, PyTorch with MPS, and MetaDrive 0.4.3. Run commands from the repository root. The compact v2 and v3 direct-action checkpoints are included; raw expert data and training checkpoints are excluded.

```bash
# Install the simulator and inference dependencies in a fresh environment.
conda create -n vision-to-action python=3.11 -y
conda activate vision-to-action
python -m pip install -r requirements-metadrive.txt

# Verify coordinate-frame, timing, speed-unit, and success-rule checks.
python -m unittest benchmarks.metadrive.test_pipeline -v

# Run the packaged camera model and a matched zero-steer baseline.
python benchmarks/metadrive/run.py --policy action \
  --checkpoint benchmarks/metadrive/weights/action_v3_dagger_best.pt \
  --seed 8000 --episodes 20 --max-steps 1000 \
  --output benchmarks/metadrive/reproduction_action.csv
python benchmarks/metadrive/run.py --policy zero \
  --seed 8000 --episodes 20 --max-steps 1000 \
  --output benchmarks/metadrive/reproduction_zero.csv
python benchmarks/metadrive/summarize_results.py \
  benchmarks/metadrive/reproduction_action.csv benchmarks/metadrive/reproduction_zero.csv
```

Recording uses the same model input resolution; a larger camera buffer can make the output video clearer without giving the model additional pixels:

```bash
python benchmarks/metadrive/run.py --policy action \
  --checkpoint benchmarks/metadrive/weights/action_v3_dagger_best.pt \
  --seed 8013 --episodes 1 --max-steps 1000 \
  --camera-width 384 --camera-height 240 \
  --record benchmarks/metadrive/demo/reproduced_seed_8013.mp4
```

On macOS with MetaDrive 0.4.3, RGB evaluation and recording require a logged-in desktop session for Panda3D's offscreen window; a physics-only headless session is not sufficient.

Training and data-preparation commands, including the expert/DAgger boundary, are documented in [`benchmarks/metadrive/README.md`](benchmarks/metadrive/README.md). The real-data trajectory experiment remains in `prepare_comma1m.py`, `train_comma1m.py`, and `train_comma1m_v2.py`.

## Limitations and next research questions

- A small steering error compounds over time; several roads still end off-road. Open-loop steering accuracy alone did not predict driving success.
- The student has no navigation command, so some intersections or route choices may be ambiguous from camera and speed alone.
- The speed controller is hand-designed. This project does not claim fully learned longitudinal control, real-world transfer, operation in arbitrary simulators, or safety readiness.
- A stronger result would require more diverse corrective data, route-conditioned planning, and a larger untouched benchmark.

## Author

Siran Liu · UBC MEng Mechatronics · BASc Manufacturing Engineering

*Independent project. Not affiliated with any company.*
