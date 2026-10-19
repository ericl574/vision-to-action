# MetaDrive closed-loop experiment

This folder contains the camera-only student, expert-label collection, trajectory-domain-shift experiments, fixed-seed evaluator, and MP4 recordings. MetaDrive's PPO expert is used **only to label training states**; `run.py --policy action` loads a PyTorch checkpoint and never calls it.

Portfolio media: [v3 final-test route, seed 8013](demo/action_v3_final_test_seed_8013.mp4), [v3 final-test route, seed 8018](demo/action_v3_final_test_seed_8018.mp4), and retained [v2 highlights](demo/action_v2_highlights.mp4). The v3 clips are clean completions selected from the 20-road final test; the complete result CSV is tracked alongside the code.

## Reproduced pipeline

1. Collect successful expert demonstrations with `collect_expert.py`. The saved RGB frame and teacher steering are captured at the same pre-step state. Some earlier local collections used a column named `speed_kmh`; despite that name, MetaDrive's `agent.speed` is in **m/s**. `prepare_actions.py` handles both old and corrected column names.
2. Build action samples with `prepare_actions.py`. Inputs are four frames at offsets −1.5, −1.0, −0.5, and 0 seconds plus speed. The target is normalized expert steering at the current image. Train/validation separation is by complete route, not frame.
3. Train the CNN-GRU with `train_actions.py`. Turn examples receive higher loss weight. The first model was trained on 71 successful routes / 9,824 indexed samples; its held-out-route steering MAE was 0.0368 versus 0.0525 for always predicting zero.
4. Collect DAgger-style data with `collect_dagger.py`. The existing student acts on camera observations, the privileged expert labels those *student-visited* states, and executed steering is an explicit student/expert mixture. The first 40-route batch produced 20,311 additional expert-labeled states. Fine-tune from the first model while reusing its original held-out-route validation split.
5. Evaluate in `run.py` with no teacher access. It records per-route success, collision, road departure, route completion, distance, and inference count. `summarize_results.py` applies the strict safe-success rule even to older result files.

Representative commands from the repository root:

```bash
python benchmarks/metadrive/collect_expert.py --episodes 30 --seed 1201
python benchmarks/metadrive/prepare_actions.py --stride 3 \
  --output data/processed/metadrive_actions_v1
python benchmarks/metadrive/train_actions.py \
  --data data/processed/metadrive_actions_v1/samples.csv \
  --output checkpoints/metadrive_actions_v1

python benchmarks/metadrive/collect_dagger.py \
  --student-checkpoint checkpoints/metadrive_actions_v1/best.pt \
  --episodes 40 --seed 1401 --beta 0.5
python benchmarks/metadrive/prepare_actions.py --stride 3 \
  --output data/processed/metadrive_actions_v2
python benchmarks/metadrive/train_actions.py \
  --data data/processed/metadrive_actions_v2/samples.csv \
  --output checkpoints/metadrive_actions_v2_dagger \
  --initial-checkpoint checkpoints/metadrive_actions_v1/best.pt \
  --split-file checkpoints/metadrive_actions_v1/split.json \
  --dagger-repeat 2 --epochs 10 --lr 0.0001
```

The full local training collection also contains older seeds 1000–1149 and additional DAgger data, so running only the commands above will create a smaller dataset and will **not** reproduce the exact packaged weights. Dataset manifests, exact split files, and checkpoint metadata are stored with each local run; raw training images and large experiment outputs are Git-ignored. Compact v2 and selected v3 checkpoints are packaged under [`weights/`](weights/) so inference does not require collecting data.

## Evaluation protocol and known limitations

- Training seeds are below 5000. Seeds 5010–5029 were used for development and the matched v2/v3 checkpoint selection; 6000–6019 were a first untouched 600-step test; 7000–7019 were a separate untouched 1,000-step v2 test; seeds 8000–8019 were reserved as the final v3/zero-baseline test and were not used for model or configuration selection.
- All routes use `map=3`, traffic density 0.05, MetaDrive 0.4.3, target speed 20 km/h, and the model's 192×120 RGB input. The recording option may use a 384×240 render buffer, downsampled to the same 192×120 student input.
- A clean success means `arrive_dest` **and** route completion ≥0.95 **and** no collision/off-road. This excludes a simulator anomaly where a car at spawn reports arrival and collision simultaneously.
- A 600-step route at 20 km/h cannot travel much beyond 333 m, so the longer 1,000-step test was reported separately. Raising target speed to 30 km/h on development roads made driving worse (0/20 success), so the final model stayed at 20 km/h.
- On development seeds 5010–5029 at 1,000 steps, v2 achieved 4/20 clean successes (mean route completion 0.437, off-road 80%, collisions 0%); v3 achieved 7/20 (0.615, off-road 60%, collisions 10%). On final seeds 8000–8019, v3 achieved 3/20 (0.545, off-road 80%, collisions 10%, timeouts 0%) versus zero steering at 1/20 (0.295, off-road 95%, collisions 10%, timeouts 0%). A clean success requires arrival, ≥95% route completion, and no collision/off-road event.
- This project currently does not use a high-level navigation instruction. Camera and speed alone may not determine the intended branch at every junction. It is a research demo, not an official MetaDrive leaderboard result or a claim of safe autonomous driving.

## Other experiment

`prepare_trajectories.py`, `train_trajectory.py`, and `model_controller.py` retain the real-camera-to-trajectory transfer experiment. The trajectory controller's steering sign and MetaDrive speed units are checked in `test_pipeline.py`. Fine-tuning the real-data checkpoint improved prediction loss, but its closed-loop driving remained inferior to the direct action model; the diagnostic metrics are preserved under this folder.
