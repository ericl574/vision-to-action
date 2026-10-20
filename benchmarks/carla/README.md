# CARLA end-to-end camera driving benchmark

Camera-only, end-to-end driving in CARLA 0.9.16 (Town10HD, Unreal Engine 4.26, Epic quality). The
student network (`PilotNet`, working name **pilot**) receives the last three front-camera frames
(512×288, 0.3 s apart), its own speed, and a navigation command (follow lane / left / right /
straight, as a car's sat-nav would give). It outputs **steering and a desired speed**. A fixed
proportional loop turns desired speed into throttle/brake. There are **no hand-written rules about
traffic lights, pedestrians or other vehicles** anywhere in the control path: if the network does not
decide to slow down, the car does not slow down.

The network also has auxiliary heads (traffic-light state, hazard ahead, hazard distance). They are
trained alongside the control heads and drawn on the overlay as "network sees", but they never feed
control.

[Watch the held-out demo reel (78 s)](demo/pilot_v2_heldout_reel.mp4): three successful routes from the
15-route held-out test set below, labeled as selected successes.

## How it was trained

| Stage | Data | What it adds |
| --- | --- | --- |
| Expert demonstrations, pass A | 60 Town10HD training routes, CARLA `BehaviorAgent` driving, seeded traffic (40 vehicles, 40 pedestrians), 4 weathers, small steering perturbations on straight roads | 61,775 labelled frames at 10 Hz |
| pilot v1 | behaviour cloning on pass A, 8 epochs | first model |
| Expert pass B | same routes (59 finished), new traffic seeds | 58,248 frames |
| DAgger round 1 | v1 drives (expert executes 40 % of steps), the expert labels every visited state | 34 routes, 34,631 frames from states the student actually reaches |
| **pilot v2** | fine-tuned from v1 on A + B + DAgger 1, 6 epochs | 68,721 training samples |
| DAgger round 2 → pilot v3 | v2 drives (expert 20 %) on 51 routes (51,716 frames), fine-tune from v2, 5 epochs | better dev completion, worse test (see results) |
| DAgger round 3 → pilot v4 | v3 drives (expert 20 %), fine-tune from v3, 4 epochs | no dev improvement (6/10); not run on test |

The expert is privileged: it reads the simulator's traffic-light state and actor positions, and it is
used **only** to drive during collection and to label states. `BehaviorAgent` notices crossing
pedestrians late, so the expert additionally brakes for any in-lane vehicle/pedestrian inside its
stopping distance (ground-truth corridor check). That privileged braking is part of the teacher, not
of the student.

Sampling oversamples decision moments: moving while the label says stop (start braking), stopped while
the label says go (start moving, the classic end-to-end "inertia" failure), and turns. Waiting at a
red light, which is most of the stopped data, is down-weighted.

## Results

All numbers are produced by `run.py` and committed under [`results/`](results/). Routes are seeded
spawn-point pairs, 300–800 m long, each containing at least one junction manoeuvre. Train, dev and test
route sets are disjoint pairs. Dev routes were used to compare checkpoints. **Test routes were run only
for the final table.**

**Success rule (strict):** the car arrives with ≥ 95 % route completion **and** has no collision, no
off-road, no route deviation, **and no red-light violation**. A red-light violation means entering a
junction's stop zone on red and driving out of it while still red. The expert is scored under the same
rule.

### Held-out test, 15 routes (Town10HD)

| Policy | Success | Mean completion | Collisions | Red-light violations | Off-road / deviation | Timeout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Expert (privileged teacher) | 11/15 | 0.963 | 7% | 3 (0.48/km) | 0% | 0% |
| pilot v1 (behaviour cloning) | 5/15 | 0.594 | 27% | 0 (0.00/km) | 40% | 0% |
| **pilot v2 (+ DAgger round 1)** | **10/15** | 0.819 | 13% | **0 (0.00/km)** | 20% | 0% |
| pilot v3 (+ DAgger round 2) | 8/15 | 0.809 | 13% | 0 (0.00/km) | 27% | 7% |

### Dev, 10 routes (Town10HD), used for checkpoint selection

| Policy | Success | Mean completion | Collisions | Red-light violations | Off-road / deviation | Timeout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Expert (privileged teacher) | 7/10 | 0.835 | 20% | 1 (0.29/km) | 0% | 0% |
| pilot v1 (behaviour cloning) | 4/10 | 0.564 | 30% | 2 (0.77/km) | 30% | 0% |
| pilot v2 (+ DAgger round 1) | 7/10 | 0.719 | 10% | 0 (0.00/km) | 20% | 0% |
| pilot v3 (+ DAgger round 2) | 7/10 | 0.818 | 20% | 1 (0.29/km) | 0% | 0% |
| pilot v4 (+ DAgger round 3) | 6/10 | 0.728 | 30% | 1 (0.32/km) | 10% | 0% |

What the numbers say, and what they don't:

- DAgger round 1 was the biggest single improvement on dev (4/10 → 7/10, completion 0.56 → 0.72,
  red-light violations 2 → 0). The early-route failures it fixed were understeering in sharp turns,
  which is the textbook covariate-shift failure DAgger addresses.
- On the held-out test, the camera-only student comes within one route of the privileged expert,
  with no red-light violations against the expert's three. Three of the expert's four failed test
  routes were red-light violations by `BehaviorAgent` under this rule; the fourth was a pedestrian collision.
  That says the teacher is imperfect under a strict rule. It does **not** mean the student is a safer
  driver in general: 15 routes is a small sample, and the student's remaining failures (two sharp
  turns leaving the lane, one wrong branch, one kerb strike, one rear-end) are more severe in kind.
- The student's mean completion (0.82) is clearly below the expert's (0.96): when it fails, it fails
  earlier in the route.
- **Checkpoint selection on 10 dev routes was not reliable.** By the dev rule (successes, then
  completion) v3 was selected over v2: same 7/10, higher completion, no off-road. On the 15 test
  routes, v3 scored 8/15 to v2's 10/15, with two pedestrian collisions where v2 had none. Both rows
  are reported, the selection was made before the test run, and the demo uses v2 and is labelled v2.
  The lesson is that a larger dev set is needed before more DAgger rounds can be judged.
- Dev route 007 ends in a collision within the first seconds for both the expert and every student (a
  vehicle crosses the spawn point). It is kept in the table.

Per-route outcomes, collision partners and distances are in each JSON file.

### Training curves (validation routes held out by route, not by frame)

| Model | Steering MAE | Desired-speed MAE | Traffic-light acc. | Hazard acc. |
| --- | ---: | ---: | ---: | ---: |
| pilot v1 (best epoch) | 0.030 | 0.72 m/s | 0.93 | 0.93 |
| pilot v2 (best epoch) | 0.023 | 0.50 m/s | 0.94 | 0.95 |

## Limitations

- One town (Town10HD), one vehicle, daytime/sunset weathers; no unseen-town evaluation yet.
- 10 dev and 15 test routes: differences of one or two routes are within noise.
- The red-light rule is a simplified stop-zone check implemented here, not the CARLA Leaderboard's
  infraction logic.
- The navigation command comes from the route planner (as sat-nav would). The student does not plan
  routes.
- No clean pedestrian-yield moment occurred in the selected test clips, so the reel does not claim
  one. Pedestrian handling is measured only through the collision column.
- The whole pipeline ran in one night on a single rented GPU (~$5 of compute), which bounded the data
  size and the number of DAgger rounds.

## Layout

| File | Role |
| --- | --- |
| `setup_box.sh` | Prepares a rented Linux GPU container: Vulkan check, CARLA download, Python deps |
| `carla_smoke.py` | Starts CARLA headless and saves one camera frame |
| `carla_env.py` | Synchronous world, ego + camera, seeded traffic/walkers, route tracking, privileged `info` |
| `routes.py`, `make_routes.py`, `routes/` | Seeded spawn-pair routes per split (committed JSON) |
| `expert.py` | BehaviorAgent wrapper with desired-speed label, hazard braking, recovery perturbations |
| `episode.py`, `metrics.py` | Episode loop, termination rules, strict success rule, red-light monitor |
| `collect.py`, `collect_parallel.sh` | Expert and DAgger collection (restart-safe, N simulators in parallel) |
| `record.py`, `dataset.py` | On-disk route format and route-split training samples |
| `model.py`, `train.py` | `PilotNet`, `PilotPolicy`, `SpeedServo`; resumable trainer |
| `run.py`, `summarize_results.py` | Closed-loop evaluation with per-step logs; README table |
| `hud.py` | Offline overlay renderer and demo reel builder |
| `sync_from_box.sh` | Pulls results, best checkpoints and videos from the GPU box |
| `weights/` | fp16 checkpoints of selected models |
| `test_carla_pipeline.py` | Simulator-free tests (`python -m unittest benchmarks.carla.test_carla_pipeline`) |

## Reproduce

```bash
bash benchmarks/carla/setup_box.sh /workspace          # once per machine (Linux + NVIDIA, Vulkan)
source /workspace/carla_env.sh
python benchmarks/carla/carla_smoke.py --out /workspace/smoke_camera.png

# expert demonstrations, 4 simulators in parallel (routes are committed under routes/)
bash benchmarks/carla/collect_parallel.sh benchmarks/carla/routes/train_town10.json /workspace/data/expert_town10 4 --max-steps 2500
bash benchmarks/carla/collect_parallel.sh benchmarks/carla/routes/train_town10.json /workspace/data/expert_town10_b 3 --max-steps 2500 --tag b --seed-offset 1000

# pilot v1
python benchmarks/carla/train.py --data /workspace/data/expert_town10 --output /workspace/runs/pilot_v1 --epochs 8 --frame-step 2

# DAgger round 1 -> pilot v2
bash benchmarks/carla/collect_parallel.sh benchmarks/carla/routes/train_town10.json /workspace/data/dagger1_town10 3 \
  --mode dagger --student-checkpoint /workspace/runs/pilot_v1/best.pt --beta 0.4 --tag d1 --seed-offset 2000 \
  --max-steps 2500 --exclude-split /workspace/runs/pilot_v1/split.json
python benchmarks/carla/train.py --data /workspace/data/expert_town10 /workspace/data/expert_town10_b /workspace/data/dagger1_town10 \
  --output /workspace/runs/pilot_v2 --initial-checkpoint /workspace/runs/pilot_v1/best.pt \
  --split-file /workspace/runs/pilot_v1/split.json --epochs 6 --lr 1e-4 --frame-step 2

# evaluate (frames + network outputs are logged for the overlay)
python benchmarks/carla/run.py --policy pilot --checkpoint benchmarks/carla/weights/pilot_v2_best_fp16.pt \
  --routes-file benchmarks/carla/routes/test_town10.json --out benchmarks/carla/results/pilot_v2_test_town10.csv \
  --log-dir /workspace/runs/eval_v2_test --launch
python benchmarks/carla/hud.py render --log /workspace/runs/eval_v2_test/test_town10_010 --out clip.mp4
```

CARLA re-runs are not bit-for-bit reproducible (traffic timing differs slightly), so a re-run gives
close but not identical per-route outcomes.
