# CARLA End-to-End Camera Driving — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (native execution chosen by the author for speed). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A camera-only, end-to-end student network trained from CARLA's expert drives held-out routes in CARLA, with an honest benchmark table and an overlay video, within 48 hours.

**Architecture:** `benchmarks/carla/` mirrors `benchmarks/metadrive/`: a thin CARLA env wrapper (sync mode, sensors, traffic, seeded routes), an expert wrapper that produces control + auxiliary labels, one shared collection loop used for both expert and DAgger collection, a CARLA-free dataset/model/trainer, an evaluator that logs frames and network outputs per route, and an offline overlay renderer. Everything that does not import `carla` is unit-tested on the Mac with synthetic data.

**Tech Stack:** CARLA 0.9.16 (`pip install carla==0.9.16`, server tarball `https://downloads.carlasim.com/Linux/CARLA_0.9.16.tar.gz`), PyTorch + torchvision (ResNet-34 pretrained), numpy, opencv-python, Python 3.10–3.12 on the Vast `vastai/pytorch` image.

**Spec:** `docs/superpowers/specs/2026-10-19-carla-end-to-end-design.md`

## Global Constraints

- No privileged state (traffic lights, walker/vehicle positions, route geometry beyond the one-hot command) reaches the student's control path at evaluation or demo time.
- Overlay shows only network outputs or elements explicitly labeled as ground truth.
- Every evaluated route appears in the results CSV/JSON; success = arrival with ≥95% completion and no collision, off-road, route deviation, or red-light violation.
- Train/held-out split by route, never by frame; held-out route lists committed under `benchmarks/carla/routes/`.
- Long jobs resumable; outputs written incrementally; weights/results/videos synced to the Mac at each milestone.
- Student placeholder name: `pilot` (author will rename later). Table rows: `<Name> v1`, `<Name> v2 (+ DAgger)`.
- Box money: ≤ $40 total; show progress at each milestone before top-ups.

## Review Focus

1. Red-light violation detection: ego crosses the stop line of a light that is red for its lane → violation recorded; a red light for a *different* lane at the same junction must not count. Test in `metrics.py` with synthetic light/lane data.
2. Route command at a junction: command must switch to `left/right/straight` *before* the junction (look-ahead), and back to `follow` after, so the student is told early enough. Test in `routes.py` with a synthetic waypoint list.
3. Speed label at a red light: desired-speed label must go to 0 while stopped, and rise again after green; a stuck-at-zero model would otherwise look "right" on the loss. Dataset balancing test checks braking frames are oversampled, not dropped.
4. Frame history on the first steps of a route: history must be filled with copies of the first frame, never zeros (MetaDrive convention). Test in `dataset.py`/policy.
5. Collector interruption: killing `collect.py` mid-route must leave a valid dataset (completed routes intact, partial route discarded on next load). Test the manifest loader with a truncated route directory.

---

## Milestones (what the author sees, in order)

| M | Deliverable | Evidence |
|---|---|---|
| M0 | Box ready: Vulkan OK, CARLA 0.9.16 runs headless | `vulkaninfo` summary + CARLA camera PNG |
| M1 | Expert drives a seeded route with traffic & walkers; labels logged | MP4 of expert + label CSV head |
| M2 | Dataset collected (≥1.5 h) across Town10HD + Town03 | manifest stats: frames, braking %, command histogram |
| M3 | `pilot v1` trained; drives held-out dev routes | loss curve + results CSV + first student MP4 |
| M4 | DAgger round(s) → `pilot v2`; dev-set selection | results CSV v1 vs v2 |
| M5 | Final held-out eval (all rows), demo video with overlay, README | table + MP4 committed |

---

### Task 1: Box setup script (M0)

**Files:**
- Create: `benchmarks/carla/setup_box.sh`
- Create: `benchmarks/carla/requirements-carla.txt`

**Produces:** `$CARLA_ROOT=/workspace/carla` with `CarlaUE4.sh`; `PYTHONPATH` includes `$CARLA_ROOT/PythonAPI/carla` (for `agents`); a `carla_smoke.py` run that saves `/workspace/smoke_camera.png`.

- [ ] Script order: `vulkaninfo --summary` (install `vulkan-tools libvulkan1` if missing; create `/usr/share/vulkan/icd.d/nvidia_icd.json` pointing at `libGLX_nvidia.so.0` if absent) → abort with clear message if no GPU listed → `pip install carla==0.9.16` + requirements → download tarball with `aria2c`/`curl -C -` to `/workspace/carla` → `apt-get install -y libomp5 xdg-user-dirs ffmpeg` → print next command.
- [ ] `benchmarks/carla/carla_smoke.py`: launch server (`-RenderOffScreen -nosound -quality-level=Epic -carla-rpc-port=2000`), connect, load `Town10HD_Opt`, spawn a vehicle + RGB camera, tick 30 frames in sync mode, save PNG, destroy actors, kill server. Prints `SMOKE OK <fps>`.
- [ ] Commit.

### Task 2: CARLA env wrapper + routes

**Files:**
- Create: `benchmarks/carla/carla_env.py` (imports carla)
- Create: `benchmarks/carla/routes.py` (no carla import; pure geometry on `(x, y, yaw, road_option)` tuples)
- Test: `benchmarks/carla/test_carla_pipeline.py::RouteTests`

**Produces:**
- `routes.py`: `RoadOption` ints `{FOLLOW=0, LEFT=1, RIGHT=2, STRAIGHT=3}`; `command_at(route_xy_opts, ego_xy, lookahead_m=25.0) -> int` returns the first non-FOLLOW option within `lookahead_m` along the route else FOLLOW; `route_progress(route_xy, ego_xy) -> (completion_frac, lateral_dist_m)`; `load_routes(path) -> list[RouteSpec]` / `save_routes` (JSON: `{town, seed, start_idx, end_idx, weather}`).
- `carla_env.py`: `class CarlaEnv` with `__init__(host, port, town, seed, traffic_vehicles, traffic_walkers, weather, image_w, image_h, fov, fixed_dt=0.05)`, `reset(route: RouteSpec) -> obs`, `step(control: (steer, throttle, brake)) -> (obs, info)`, `close()`. `obs = {"rgb": HxWx3 uint8, "speed_mps": float, "command": int}`. `info` has privileged fields for labels/metrics only: `tl_state` (0 none/1 green/2 yellow/3 red for ego lane), `hazard` (bool), `hazard_dist_m`, `collision_type` (None/"vehicle"/"walker"/"static"), `off_road` (bool), `ran_red_light` (bool, edge-triggered), `route_completion`, `lateral_dev_m`, `arrived`, `ego_xy`.
- Red-light detection: uses `vehicle.is_at_traffic_light()`/`get_traffic_light()` per tick plus stop-line crossing (ego leaves the light's trigger volume while state is Red and speed > 1 m/s).
- Route generation: `generate_routes(town, seed, n, min_m, max_m) -> list[RouteSpec]` using `GlobalRoutePlanner(map, 2.0)`, filtered to routes that contain ≥1 junction and no `CHANGELANELEFT/RIGHT` options.

- [ ] Write `RouteTests`: `test_command_switches_before_junction` (synthetic straight route with a LEFT at 40 m: at 10 m → FOLLOW, at 20 m → LEFT, after the junction → FOLLOW); `test_route_progress_monotonic`; `test_routes_roundtrip_json`.
- [ ] Run: `python -m unittest benchmarks.carla.test_carla_pipeline.RouteTests -v` → FAIL; implement `routes.py`; → PASS.
- [ ] Implement `carla_env.py` (box-tested in Task 3). Commit.

### Task 3: Expert wrapper + shared collection loop (M1)

**Files:**
- Create: `benchmarks/carla/expert.py`
- Create: `benchmarks/carla/collect.py`
- Create: `benchmarks/carla/record.py` (frame/label writer, no carla import)
- Test: `test_carla_pipeline.py::RecordTests`

**Produces:**
- `expert.py`: `class Expert` wrapping `BehaviorAgent(vehicle, behavior="normal")` with `set_route(route_waypoints)`, `act(info) -> (steer, throttle, brake, target_speed_mps)`; `target_speed_mps` is the agent's local-planner target speed, or 0 when the agent emergency-brakes (`_affected_by_traffic_light` / hazard managers). Steering perturbation: with prob `p=0.1` per second start a 1–2 s triangular steer offset of ±0.15–0.3; frames during perturbation keep the *expert's* steer label.
- `record.py`: `class RouteRecorder(root, route_id)`: `append(rgb, row: dict)` writes JPEG `frames/{idx:06d}.jpg` (quality 90) + buffers rows; `finish(status)` writes `labels.csv` and `meta.json` (`status`, `n_frames`, `town`, `seed`); rows: `idx, t, speed_mps, command, steer, throttle, brake, target_speed_mps, tl_state, hazard, hazard_dist_m, is_perturbed, student_steer, student_target_speed, beta_used_expert`. `load_manifest(root) -> list[RouteMeta]` skips directories without `meta.json`.
- `collect.py`: `--mode expert|dagger --town --routes-file --n-routes --seed --out --max-steps --beta --student-checkpoint --port`. Shared loop at 10 Hz (two 0.05 s ticks per control). Expert mode executes expert control; DAgger mode executes `expert if rand()<beta else student`, always logs expert labels. Route ends on arrival/collision/off-road/deviation/timeout; `meta.status` records which. Parallel: multiple processes on different `--port`s.

- [ ] `RecordTests`: `test_finish_writes_labels_and_meta`; `test_manifest_skips_partial_route` (dir with frames but no meta.json ignored).
- [ ] Run → FAIL; implement `record.py` → PASS. Commit.
- [ ] Implement `expert.py`, `collect.py`. Box smoke: `collect.py --mode expert --town Town10HD_Opt --n-routes 1 --max-steps 600`, render MP4 of frames with `hud.py --no-net` (ground-truth-labeled overlay) → **M1 evidence**. Commit.

### Task 4: Dataset + model + trainer (CARLA-free)

**Files:**
- Create: `benchmarks/carla/dataset.py`, `benchmarks/carla/model.py`, `benchmarks/carla/train.py`
- Test: `test_carla_pipeline.py::DatasetTests`, `::ModelTests`

**Produces:**
- `dataset.py`: `build_samples(manifest_roots, history=3, stride=5, split_file=None, val_frac=0.15) -> (train_rows, val_rows, split)`; split by route id; `class DriveSet(Dataset)` returns `images [T,3,H,W] float`, `speed [1]`, `command int`, `targets {steer, target_speed, tl_state, hazard, hazard_dist}`; `make_sampler(rows)` weights: braking frames (`target_speed < 1.0`) ×4, turning (`|steer|>0.1`) ×3, red-light (`tl_state==3`) ×3. Speed input dropout handled in the model.
- `model.py` (no carla, no file I/O beyond torch): `class PilotNet(nn.Module)`: per-frame ResNet-34 trunk (pretrained flag), GRU over T frames (hidden 256), measurement MLP (speed with `speed_dropout=0.3` during training), command-conditioned branches: 4 heads each outputting `[steer, target_speed_norm]`; aux heads on the shared feature: `tl_logits[4]`, `hazard_logit[1]`, `hazard_dist_norm[1]`. `forward(images, speed, command) -> dict`. `class PilotPolicy(checkpoint, device)`: `reset(frame)`, `act(frame_rgb, speed_mps, command, step) -> (steer, throttle, brake, outputs)`, with `TemporalFrameHistory`-style history (reuse `benchmarks/metadrive/model_controller.TemporalFrameHistory`) and a P-servo `throttle/brake = clip(k*(target_speed - speed))`, `k=0.5`, brake when error < −0.5 m/s.
- `train.py`: resumable (`--resume`), AdamW, cosine LR, AMP, `--epochs --lr --batch-size --initial-checkpoint --split-file --data <roots...> --output`; loss = `L1(steer)*1.0 + L1(speed)*1.0 + CE(tl)*0.5 + BCE(hazard)*0.5 + L1(dist)*0.2`; logs `history.csv`, saves `best.pt` (val control loss) + `last.pt` with `{"model_state_dict", "epoch", "config", "val"}`.

- [ ] `DatasetTests`: `test_split_is_by_route`; `test_history_pads_with_first_frame`; `test_sampler_upweights_braking`. `ModelTests`: `test_forward_shapes_cpu` (B=2,T=3,H=144,W=256, pretrained=False); `test_policy_servo_brakes_when_target_below_speed`.
- [ ] Run → FAIL; implement → PASS. Commit.
- [ ] Box: `train.py` on the M1 smoke data for 1 epoch to validate the loop end-to-end. Commit.

### Task 5: Evaluator + metrics + overlay (M3 tooling)

**Files:**
- Create: `benchmarks/carla/run.py`, `benchmarks/carla/metrics.py`, `benchmarks/carla/hud.py`, `benchmarks/carla/summarize_results.py`
- Test: `test_carla_pipeline.py::MetricsTests`, `::HudTests`

**Produces:**
- `metrics.py`: `route_outcome(events: dict) -> dict` applies the success rule; `summarize(rows) -> dict` (success rate, mean completion, per-type collision rates, red-light violations per km, deviation/off-road/timeout rates). Pure functions.
- `run.py`: `--policy expert|pilot --checkpoint --routes-file --out results/<name>.csv/json --log-frames <dir> --max-steps --port`. For `pilot`, the only inputs to the policy are `obs["rgb"]`, `obs["speed_mps"]`, `obs["command"]`. Logs per-step `outputs` (steer, target_speed, tl_probs, hazard_prob, hazard_dist) + frame JPEG for overlay.
- `hud.py`: `render(frames_dir, log_csv, out_mp4, fps=10, net=True)`: dark panel with steering arc, speed vs desired speed, command arrow, "network sees" bars (light state probs, hazard prob + dist), one-line caption from outputs only (`desired 0.0 m/s → braking`), small "GT" badges only when `--no-net`. `make_reel(clips: list[(mp4, start_s, end_s, caption)], out)` concatenates with captions via ffmpeg.
- `summarize_results.py`: prints the README markdown table from one or more results CSVs.

- [ ] `MetricsTests`: `test_red_light_disqualifies_success`; `test_other_lane_red_not_counted` (event only when `tl_state_for_ego==3`); `test_success_requires_95_completion`. `HudTests`: `test_render_single_frame_shape` (synthetic frame + outputs → image HxW matches).
- [ ] Run → FAIL; implement → PASS. Commit.

### Task 6: Data collection at scale (M2)

- [ ] Generate and commit route files: `routes/train_town10.json` (60), `routes/train_town03.json` (60), `routes/dev_town10.json` (10), `routes/dev_town03.json` (10), `routes/test_town10.json` (15), `routes/test_town03.json` (15), `routes/test_town02.json` (10, unseen), seeds documented in README. Weather sampled from `{ClearNoon, CloudyNoon, WetNoon, SoftRainSunset}`.
- [ ] Launch 3 CARLA servers (ports 2000/2010/2020) + 3 `collect.py --mode expert` in `tmux`, log to `/workspace/logs/`. Target ≥ 50k frames at 10 Hz.
- [ ] Report manifest stats (frames, braking %, red-light %, command histogram) → **M2 evidence**.

### Task 7: pilot v1 (M3)

- [ ] Train BC on collected data (`--epochs 15`), copy `best.pt` + `history.csv` to Mac.
- [ ] `run.py --policy expert` and `--policy pilot` on dev routes → results CSVs; first student MP4 via `hud.py` → **M3 evidence**. Commit results + weights (fp16, `.gitignore` exception like MetaDrive).

### Task 8: DAgger → pilot v2 (M4)

- [ ] `collect.py --mode dagger --beta 0.5` with v1 on train routes (≥ 20k frames), then `--beta 0.2`; fine-tune from v1 with original split file.
- [ ] Dev-set eval v1 vs v2; select checkpoint → **M4 evidence**.

### Task 9: Final evaluation, demo, README (M5)

- [ ] Final eval on untouched test routes: expert, v1, v2 (+ Town02 row if time). Second box allowed for this.
- [ ] Pick 2–4 clips from logged test runs (red stop, left, right, pedestrian yield); `make_reel` → `demo/pilot_v2_heldout_reel.mp4` (≤ 25 MB).
- [ ] `benchmarks/carla/README.md` + root README section: what the student sees, table, limitations, reproduce commands. Commit, push, sync to Mac.
