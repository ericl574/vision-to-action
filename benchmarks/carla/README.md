# CARLA end-to-end camera driving benchmark

Camera-only, end-to-end driving in CARLA 0.9.16. The student network (`PilotNet`, working name
`pilot`) receives a short history of front-camera frames, its own speed and a navigation command
(follow / left / right / straight), and outputs steering and a desired speed. A fixed proportional
speed loop turns desired speed into pedal commands. There are no hand-written rules about traffic
lights, pedestrians or other vehicles anywhere in the control path. Auxiliary heads (traffic-light
state, hazard ahead) are training signal and overlay content only.

CARLA's `BehaviorAgent` is the privileged teacher: it drives during demonstration collection and
labels every state the student visits during DAgger rounds. It is never called at evaluation time.

Status: **work in progress** (started 2026-10-19). Results tables are filled in from the committed
CSV/JSON files under `results/` as runs finish; nothing here is a claim until a file backs it.

## Layout

| File | Role |
| --- | --- |
| `setup_box.sh` | Prepares a rented Linux GPU container: Vulkan check, CARLA download, Python deps |
| `carla_smoke.py` | Starts CARLA headless and saves one camera frame |
| `carla_env.py` | Synchronous world, ego + camera, seeded traffic/walkers, route tracking, privileged `info` |
| `routes.py`, `make_routes.py`, `routes/` | Seeded spawn-pair routes per split (committed JSON) |
| `expert.py` | BehaviorAgent wrapper with desired-speed label and recovery perturbations |
| `episode.py`, `metrics.py` | Episode loop, termination rules, strict success rule, red-light monitor |
| `collect.py`, `collect_parallel.sh` | Expert and DAgger collection (restart-safe) |
| `record.py`, `dataset.py` | On-disk route format and route-split training samples |
| `model.py`, `train.py` | `PilotNet`, `PilotPolicy`, `SpeedServo`; resumable trainer |
| `run.py`, `summarize_results.py` | Closed-loop evaluation with per-step logs; README table |
| `hud.py` | Offline overlay renderer and demo reel builder |
| `test_carla_pipeline.py` | Simulator-free tests (`python -m unittest benchmarks.carla.test_carla_pipeline`) |

## Success rule

A route counts as a success only if the car arrives with ≥ 95 % route completion **and** had no
collision, no off-road event, no route deviation **and no red-light violation**. Red-light violations
are also reported in their own column. The expert row is measured under the same rule.

## Reproduce (box side)

```bash
bash benchmarks/carla/setup_box.sh /workspace          # once per machine
source /workspace/carla_env.sh
python benchmarks/carla/carla_smoke.py --out /workspace/smoke_camera.png

# routes (already committed under routes/; regenerate only to change the splits)
python benchmarks/carla/make_routes.py --town Town10HD_Opt --n 60 --seed 100 --out benchmarks/carla/routes/train_town10.json --launch

# expert demonstrations, 3 simulators in parallel
bash benchmarks/carla/collect_parallel.sh benchmarks/carla/routes/train_town10.json /workspace/data/expert_town10 3

# pilot v1
python benchmarks/carla/train.py --data /workspace/data/expert_town10 /workspace/data/expert_town03 --output /workspace/runs/pilot_v1

# evaluate + log frames for the overlay
python benchmarks/carla/run.py --policy pilot --checkpoint /workspace/runs/pilot_v1/best.pt \
  --routes-file benchmarks/carla/routes/dev_town10.json --out benchmarks/carla/results/pilot_v1_dev_town10.csv \
  --log-dir /workspace/runs/eval_pilot_v1_dev_town10 --launch
python benchmarks/carla/hud.py render --log /workspace/runs/eval_pilot_v1_dev_town10/dev_town10_003 --out demo/clip.mp4
```

## Results

_To be filled from `results/*.csv` as evaluations complete._

## Limitations

_To be written with the final results._
