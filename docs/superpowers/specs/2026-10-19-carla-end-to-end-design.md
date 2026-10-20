# CARLA end-to-end camera driving demo — design

Date: 2026-10-19. Deadline: evening of 2026-10-21 (about 48 hours).

## Intent

Produce a portfolio piece for a Tesla Autopilot/AI internship application: a
neural network that the author trained drives a car in a photorealistic
simulator (CARLA) from camera images alone, handling traffic lights, left and
right turns at intersections, and pedestrians, with an FSD-style on-screen
overlay that shows what the network is outputting and why the car is doing what
it does.

The author chose **fully end-to-end** control: the network itself outputs the
driving decision (steering and desired speed). There are no hand-written rules
about traffic lights, pedestrians, or other vehicles anywhere in the control
path. If the network does not learn to stop, the car does not stop. This mirrors
Tesla FSD v12+ and is the story the author wants to tell.

The project extends the existing repository, which already does the same thing
in MetaDrive (camera-only CNN-GRU student, privileged expert as teacher,
behavior cloning + DAgger, fixed-seed benchmark with honest results tables).
The CARLA work lives in `benchmarks/carla/` beside `benchmarks/metadrive/` and
follows the same conventions: expert used only to label, student never sees
privileged state at inference, results CSV/JSON committed, README that states
real success rates rather than claims.

## Goals

1. **End-to-end policy.** Input: front camera image(s), current speed, and a
   high-level route command (follow lane / left / right / straight at the next
   junction). Output: steering and desired speed. A simple speed servo (as in
   the MetaDrive project) converts desired speed into throttle/brake; it is an
   actuator loop, not a decision maker.
2. **Trained by the author's pipeline** on data from CARLA's built-in expert
   agent (which obeys lights and yields to walkers), then improved with at
   least one DAgger round focused on the states the student actually visits.
3. **Demo video, 60–90 s**, of the student driving held-out routes with the
   overlay: camera view, steering, speed vs. desired speed, route command,
   and the network's auxiliary perception outputs (traffic-light state,
   hazard ahead). Clips are selected successes and are labeled as such.
4. **Benchmark table on held-out routes** the student never trained on:
   route completion, success rate, collisions (pedestrian / vehicle / static),
   red-light violations, off-road, timeouts. Rows: expert (upper bound),
   behavior cloning only, behavior cloning + DAgger. If time allows, a row for
   an unseen town.
5. **Reproducible README** in `benchmarks/carla/` with setup, collection,
   training, evaluation, and recording commands, plus the honest limitations
   section, in the voice of the existing READMEs.

## Non-goals

- CARLA Leaderboard submission or comparison against published leaderboard
  numbers.
- LiDAR, radar, HD maps, or any sensor other than RGB cameras + speedometer.
- Generalization across many towns and weathers; one or two training towns,
  held-out routes, and at most one unseen town is enough.
- A live, interactive dashboard. The overlay is rendered into the video.
- Perfect reliability. "Handles these situations most of the time on unseen
  routes" is the bar, stated with numbers.

## Anti-goals (would make this a failure even if it technically works)

- **Hidden rules.** Any code that reads traffic-light state, pedestrian
  positions, or other privileged simulator state and uses it to change the
  student's control at evaluation/demo time. Privileged state may be read
  only for (a) expert labels during collection, (b) metrics, and (c) the
  overlay's clearly labeled "ground truth" elements, if any.
- **Misleading overlay.** Perception shown on screen must come from the
  network's own outputs, or be labeled as simulator ground truth. Nothing
  implies a detector that does not exist.
- **Cherry-picked numbers.** The video may show selected successes; the
  results table must report every evaluated route.
- **Spending the rented GPU time before the code is ready.** All code is
  written and dry-run tested locally first; the box is rented when the setup
  script and collector are ready to run.

## Constraints

- **Time: ~48 hours**, including setup, collection, training, DAgger,
  evaluation, video, README.
- **Compute:** one rented Vast.ai RTX 4090 (24 GB), 32-core host, ~130 GB RAM,
  100 GB container disk, Ubuntu 22.04, `vastai/pytorch` image. It is a
  **container, not a VM**: no Docker-in-Docker, no driver installs. CARLA must
  run as a plain binary with `-RenderOffScreen`. Vulkan passthrough is not
  guaranteed on every host; the setup script checks it first and the author
  switches hosts if it fails.
- **Author's machine:** Apple Silicon Mac, 8 GB RAM. CARLA cannot run on it.
  It is used for writing code, reviewing videos, and the Git repo. The CARLA
  Python package has no macOS wheel, so CARLA-touching code is tested on the
  box; everything else (dataset, model, training loop, overlay renderer,
  metrics) must be unit-testable on the Mac with synthetic data.
- **Money:** total ceiling about **$40 USD** (author-approved on 2026-10-19).
  First deposit is $5 (≈9 box-hours); the author wants visible progress
  (CARLA screenshot → expert video → dataset → first trained student driving)
  before topping up, and will then top up to cover unattended overnight
  running. A second concurrent instance for a few hours on day 2 is allowed if
  it is clearly useful (evaluation while the main box trains). All long jobs
  must be resumable and write outputs incrementally. Weights, results, and
  videos are synced to the Mac at every milestone; the raw frame dataset lives
  only on the box and is treated as regenerable, so collection must be cheap
  to rerun (seeded routes, scripted launch).
- **Model name:** the author will choose a product-style name for the student
  (not "behavior cloning"). Code and README use a placeholder name until then;
  table rows are "<Name> v1" and "<Name> v2 (+ DAgger)", with the technique
  explained once in prose.
- **Python version:** CARLA 0.9.16 (Sept 2025) ships PyPI wheels for Python
  3.10–3.12 on glibc ≥ 2.31, which covers the Vast PyTorch image on Ubuntu
  22.04. Use `pip install carla==0.9.16` and the matching
  `CARLA_0.9.16.tar.gz` server package; the `agents` package comes from the
  tarball's `PythonAPI/carla/agents`, not the wheel. Create a venv only if the
  image's Python falls outside 3.10–3.12.
- **Access to the box:** the builder has direct SSH access from the moment the
  instance is running (the author creates the instance in the Vast UI with the
  official PyTorch template and pastes the SSH command). Box-side debugging is
  done by the builder, not relayed through the author. CARLA-touching modules
  are therefore smoke-tested on the box; the Mac tests cover everything else.
  The setup script must verify Vulkan (`libvulkan1`, an NVIDIA ICD under
  `/usr/share/vulkan/icd.d/`, `vulkaninfo --summary` listing the GPU) **before**
  downloading the 16+ GB server package.

## Decided design

### Simulator and scenario

- CARLA 0.9.15 (or the newest 0.9.x whose Python wheel installs on the box),
  synchronous mode, fixed time step 0.05 s (20 Hz), control and recording at
  10 Hz.
- Training towns: `Town10HD` (most photorealistic; used for the demo video) and
  `Town03` (varied junctions). Held-out evaluation: routes in those towns that
  were not used for training, plus `Town02` as the unseen-town row if time
  allows.
- Routes: random start/end spawn pairs through CARLA's global route planner,
  length 300–800 m, seeded so every split is reproducible and committed as a
  route list. Train / held-out split is by route, never by frame.
- Background traffic: CARLA Traffic Manager vehicles and walkers, moderate
  density, seeded. Walkers cross roads so yielding actually occurs.
- Weather: a small set of presets (clear noon, cloudy, wet road, sunset)
  sampled per route for robustness and visual variety.

### Expert (teacher)

- CARLA's `BehaviorAgent` from `agents/navigation` following the planned route.
  It stops for red lights and for vehicles/walkers ahead. Used **only** to
  produce labels and to drive during collection.
- During expert collection, small steering perturbations are injected with
  recovery (the standard trick) so the dataset contains off-center states.
- Per-frame labels recorded: expert steering, expert desired speed (target
  speed the agent is tracking, 0 when stopping), route command, ego speed, and
  auxiliary labels from privileged state: traffic-light state affecting the ego
  (none / green / yellow / red), hazard-ahead flag (vehicle or walker within
  the agent's braking envelope), hazard distance.

### Student input

- One wide front RGB camera (FOV ~100°), captured at a larger resolution for
  video and downscaled to roughly 512×288 for the network. A short frame
  history (2–4 frames) as in the MetaDrive student, so motion is visible.
  Builder may add a center-crop "telephoto" stream if traffic lights are too
  small at that resolution.
- Current speed (with input dropout during training to counter the
  "stay stopped forever" inertia failure common to end-to-end models).
- Route command, one-hot over {follow, left, right, straight}, applied as a
  command-conditioned head (one output branch per command) as in conditional
  imitation learning.

### Student network

- ImageNet-pretrained ResNet-34 (or similar small pretrained backbone) encoder
  per frame, temporal fusion (GRU or concatenation), measurement fusion, then
  heads:
  - **Control heads (used to drive):** steering ∈ [−1, 1]; desired speed ≥ 0.
  - **Auxiliary heads (training signal and overlay only, never control):**
    traffic-light state (4-way), hazard ahead (binary), hazard distance.
- Loss: weighted sum; braking and turning frames are oversampled or upweighted
  so the rare events are learned.

### Control at runtime

- Steering goes straight to the vehicle.
- Desired speed → throttle/brake through a proportional/PI speed servo with
  fixed gains. This is the same structure as the MetaDrive project. The servo
  never reads anything but the network's desired speed and the speedometer.

### Training procedure

1. Behavior cloning on the first expert dataset (target ≥ 1.5 h of driving,
   ~50k frames at 10 Hz) — "BC".
2. DAgger: the student drives, the expert labels every visited state, executed
   control is a student/expert mixture (β schedule); aggregate; fine-tune.
   At least one round; two if time allows — "BC + DAgger".
3. Model selection on a development route set, final numbers on an untouched
   held-out route set (same discipline as the MetaDrive seeds).

### Evaluation

- `run.py --policy {expert,student} --routes <split> --checkpoint …` drives
  each route with no teacher access for the student, records per-route:
  route completion, success, collisions by type, red-light violations,
  off-road, route deviation, timeout, distance, time.
- **Route termination and success rule (builder's call, matching the
  MetaDrive convention):** a route ends at the first collision, off-road
  event, route deviation (ego farther than a fixed threshold from the planned
  route, e.g. wrong branch at a junction), arrival, or timeout. **Success**
  requires arrival with ≥95% route completion **and** no collision, no
  off-road, no route deviation, **and no red-light violation**. Red-light
  violations are recorded as their own column and are also disqualifying,
  because a "success" that admits running a red light is the first thing a
  reviewer will probe. The expert row is measured under the same rule, so its
  number is the honest upper bound for this setup, not a Leaderboard figure.
- Evaluation frames and network outputs are logged for every route at a
  moderate resolution, so demo clips come from exactly the runs in the table
  (CARLA re-runs are not bit-reproducible).
- Outputs CSV + JSON per run; a summarizer produces the README table.
- Baselines: expert (upper bound), BC, BC + DAgger, plus "constant steering 0
  / constant speed" only if cheap.

### Overlay / video

- Rendered offline from logged frames and logged network outputs (not live),
  so the drive itself is never slowed by drawing.
- Shows: camera frame; steering indicator; current speed and network desired
  speed; route command arrow; auxiliary outputs as labeled bars
  ("network: light RED 0.92", "network: hazard 0.81, 7 m"); a one-line
  decision caption derived only from network outputs ("desired speed 0 → braking").
- Optional if time allows: a Grad-CAM attention heatmap to show where the
  network is looking (the closest honest stand-in for "object detection").
- Final video: 60–90 s, 2–4 held-out clips covering a red light stop, a left
  turn, a right turn, and a pedestrian yield, with a caption stating clips are
  selected successes and pointing to the full table.

### Repository layout

```
benchmarks/carla/
  README.md            # setup, commands, results table, limitations
  setup_box.sh         # Vast box: Vulkan check, CARLA download, venv, deps
  carla_env.py         # server launch, sync mode, sensors, traffic, routes
  expert.py            # BehaviorAgent wrapper + label extraction + perturbation
  collect.py           # expert collection and DAgger collection (shared loop)
  dataset.py           # frame-history samples, balancing, split by route
  model.py             # network + policy wrapper (no CARLA import)
  train.py             # resumable training, checkpoints, curves
  run.py               # evaluation / recording driver, metrics CSV+JSON
  metrics.py           # success rules, summarizer
  hud.py               # offline overlay renderer (no CARLA import)
  routes/              # committed seeded route lists per split
  results/             # committed CSV/JSON
  demo/                # final MP4(s), small
  weights/             # final compact checkpoint
  test_carla_pipeline.py  # Mac-runnable tests with synthetic data
```

### Schedule (builder's plan; checkpoints the author will see)

Day 1 (today → tonight)
- Code written and dry-run tested on the Mac before renting.
- Rent box → `setup_box.sh` → Vulkan check → CARLA screenshot (≈1 h).
- Expert driving video with first overlay (≈3 h).
- Expert collection running in the background across both towns (≈6 h mark:
  tens of thousands of frames).
- BC training starts on the partial dataset; first student drive recorded.

Day 2
- DAgger round(s), development-set selection, final held-out evaluation.
- Demo clips, overlay polish, README with tables and limitations.
- Commit results, weights, videos; copy to Mac; push.

## Left to the builder

- Exact CARLA version, Python/venv arrangement, camera resolution and FOV,
  frame-history length, backbone choice, loss weights, learning rate, batch
  size, β schedule, traffic density, number of routes per split, weather set.
- Overlay visual design, within the constraint that it shows only network
  outputs or clearly labeled ground truth.
- Whether to add the telephoto crop, Grad-CAM, or the unseen-town row, based
  on time and on how the first student performs.
- How collection is parallelized (multiple CARLA servers on the 32-core box).

## Risks and fallbacks

- **Vulkan not available on the host** → destroy instance, rent the next
  candidate; if two hosts fail, fall back to MetaDrive with the same overlay
  and a stronger model (the author has agreed to this fallback).
- **Student runs red lights after BC** → expected; DAgger on junction
  approaches plus oversampling of braking frames is the planned fix. If still
  weak, report it honestly in the table rather than adding a rule.
- **Time** → the demo video and honest table take priority over the
  unseen-town row and Grad-CAM.
