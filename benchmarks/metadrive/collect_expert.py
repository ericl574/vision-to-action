"""MetaDrive 0.4.3 Expert -> camera/action imitation-learning demonstrations.

Run from your vision-to-action repository root:
  python benchmarks/metadrive/collect_expert.py --episodes 20 --seed 1000

The expert has privileged LiDAR/state observations. Only RGB, ego speed,
expert actions, and episode metadata are saved for the student model.
One frame corresponds to the expert action chosen BEFORE env.step(action).
"""

import argparse
import csv
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


def rgb_from_observation(obs):
    """MetaDrive image observation is H,W,3,stack in BGR uint8 order."""
    if not isinstance(obs, dict) or "image" not in obs:
        raise ValueError("Expected MetaDrive image observation dict")
    image = np.asarray(obs["image"])
    if image.ndim == 4:
        image = image[..., -1]
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Unexpected image shape {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 pixels (norm_pixel=False), got {image.dtype}")
    # MetaDrive's BaseCamera perceives BGR, and PIL requires RGB.
    return np.ascontiguousarray(image[..., ::-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000,
                        help="Training seeds. Reserve 5000+ for your existing benchmark.")
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--traffic-density", type=float, default=0.05)
    parser.add_argument("--map-blocks", type=int, default=3)
    parser.add_argument("--quality", type=int, default=80, help="JPEG quality")
    parser.add_argument("--perturb-interval", type=int, default=0,
                        help="If positive, inject a brief steering disturbance every N steps for recovery demos")
    parser.add_argument("--perturb-duration", type=int, default=4)
    parser.add_argument("--perturb-steer", type=float, default=0.12)
    parser.add_argument("--min-free-gb", type=float, default=2.0,
                        help="Abort gracefully before using the last free GBs")
    parser.add_argument("--output", type=Path,
                        default=Path("data/collected/metadrive_expert_v1"))
    args = parser.parse_args()
    if args.episodes < 1 or args.max_steps < 1 or args.map_blocks < 1:
        parser.error("episodes, max-steps and map-blocks must be >= 1")
    if not 0 <= args.traffic_density <= 1:
        parser.error("traffic-density must be between 0 and 1")
    if not 20 <= args.quality <= 100:
        parser.error("quality must be between 20 and 100")
    if args.perturb_interval < 0 or args.perturb_duration < 1 or not 0 <= args.perturb_steer <= 1:
        parser.error("invalid perturbation settings")
    if args.seed <= 0 or args.seed + args.episodes > 5000:
        parser.error("This script reserves seed 5000+ for benchmark evaluation. Use training seeds 1..4999.")

    # Separate runs, no overwrites, and exact configuration recorded.
    suffix = "_recovery" if args.perturb_interval else ""
    run = args.output / f"seed_{args.seed}_n{args.episodes}_blocks{args.map_blocks}{suffix}"
    if run.exists():
        raise FileExistsError(f"Run already exists: {run}. Use a new seed or --output.")
    if shutil.disk_usage(Path.cwd()).free < args.min_free_gb * 1024**3:
        raise RuntimeError("Not enough disk space; free storage before data collection.")

    from metadrive import MetaDriveEnv
    from metadrive.component.sensors.rgb_camera import RGBCamera
    # Deterministic reference policy, independent of student torch/MPS.
    from metadrive.examples.ppo_expert.numpy_expert import expert
    import metadrive

    config = dict(
        use_render=False,
        image_observation=True,
        image_on_cuda=False,
        norm_pixel=False,
        stack_size=1,
        sensors={"rgb_camera": (RGBCamera, 192, 120)},
        vehicle_config={"image_source": "rgb_camera"},
        start_seed=args.seed,
        num_scenarios=args.episodes,
        map=args.map_blocks,
        traffic_density=args.traffic_density,
        horizon=args.max_steps,
    )
    run.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        environment="MetaDrive",
        metadrive_version=getattr(metadrive, "__version__", "0.4.3 (verify locally)"),
        policy="metadrive.examples.ppo_expert.numpy_expert.expert(deterministic=True)",
        expert_input="privileged lidar and ego state (teacher only)",
        student_input="RGB camera + ego speed (no privileged teacher data)",
        color_order_saved="RGB",
        image_width=192,
        image_height=120,
        image_format="JPEG",
        image_quality=args.quality,
        speed_unit="m/s (MetaDrive agent.speed)",
        seed_start=args.seed,
        seed_end=args.seed + args.episodes - 1,
        max_steps=args.max_steps,
        traffic_density=args.traffic_density,
        map_blocks=args.map_blocks,
        benchmark_seeds_reserved="5000+",
        perturb_interval_steps=args.perturb_interval,
        perturb_duration_steps=args.perturb_duration if args.perturb_interval else 0,
        perturb_steering_amplitude=args.perturb_steer if args.perturb_interval else 0,
        steering_label="unperturbed expert action at the current image",
    )
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summary_fields = ["seed", "steps", "frames_saved", "success", "collision", "offroad", "truncated", "reward", "seconds"]
    action_fields = ["seed", "step", "image", "steering", "acceleration", "speed_mps", "reward",
                     "executed_steering", "perturbed",
                     "position_x", "position_y", "velocity_x", "velocity_y", "heading_theta"]
    env = None
    summaries = []
    try:
        env = MetaDriveEnv(config)
        with (run / "episodes.csv").open("w", newline="") as summary_file:
            summary_writer = csv.DictWriter(summary_file, fieldnames=summary_fields)
            summary_writer.writeheader()
            for n in range(args.episodes):
                if shutil.disk_usage(Path.cwd()).free < args.min_free_gb * 1024**3:
                    print("Disk space threshold reached. Stopping cleanly.")
                    break
                seed = args.seed + n
                episode_dir = run / f"episode_{seed}"
                frame_dir = episode_dir / "frames"
                frame_dir.mkdir(parents=True)
                started = time.perf_counter()
                observation, _reset_info = env.reset(seed=seed)
                reward_total = 0.0
                perturb_rng = np.random.default_rng(seed)
                perturb_sign = 1.0
                steps = 0
                latest_info = {}
                terminated = truncated = False
                with (episode_dir / "actions.csv").open("w", newline="") as action_file:
                    action_writer = csv.DictWriter(action_file, fieldnames=action_fields)
                    action_writer.writeheader()
                    for t in range(args.max_steps):
                        if t % 50 == 0 and shutil.disk_usage(Path.cwd()).free < args.min_free_gb * 1024**3:
                            raise RuntimeError(f"Disk threshold reached during seed {seed}; completed runs are preserved.")
                        rgb = rgb_from_observation(observation)
                        action = np.asarray(expert(env.agent, deterministic=True), dtype=np.float32).reshape(-1)
                        if action.shape != (2,) or not np.isfinite(action).all():
                            raise ValueError(f"Invalid expert action at seed {seed}, step {t}: {action}")
                        action = np.clip(action, -1.0, 1.0)
                        perturbed = bool(args.perturb_interval and t >= args.perturb_interval and
                                         t % args.perturb_interval < args.perturb_duration)
                        if perturbed and t % args.perturb_interval == 0:
                            perturb_sign = float(perturb_rng.choice([-1, 1]))
                        executed = action.copy()
                        if perturbed:
                            executed[0] = np.clip(executed[0] + perturb_sign * args.perturb_steer, -1.0, 1.0)
                        image_rel = f"frames/{t:06d}.jpg"
                        Image.fromarray(rgb).save(frame_dir / f"{t:06d}.jpg", quality=args.quality)
                        speed = float(env.agent.speed)
                        position = np.asarray(env.agent.position, dtype=np.float64)
                        velocity = np.asarray(env.agent.velocity, dtype=np.float64)
                        heading = float(env.agent.heading_theta)
                        # Save image and action at the same pre-step state.
                        next_obs, reward, terminated, truncated, latest_info = env.step(executed.tolist())
                        reward_total += float(reward)
                        action_writer.writerow(dict(
                            seed=seed, step=t, image=image_rel,
                            steering=f"{action[0]:.7f}", acceleration=f"{action[1]:.7f}",
                            speed_mps=f"{speed:.4f}", reward=f"{float(reward):.5f}",
                            executed_steering=f"{executed[0]:.7f}", perturbed=int(perturbed),
                            position_x=f"{position[0]:.6f}", position_y=f"{position[1]:.6f}",
                            velocity_x=f"{velocity[0]:.6f}", velocity_y=f"{velocity[1]:.6f}",
                            heading_theta=f"{heading:.8f}",
                        ))
                        steps += 1
                        observation = next_obs
                        if terminated or truncated:
                            break
                collision = bool(any(latest_info.get(k, False) for k in
                                     ("crash", "crash_vehicle", "crash_object", "crash_building", "crash_human")))
                summary = dict(
                    seed=seed, steps=steps, frames_saved=steps,
                    success=int(bool(latest_info.get("arrive_dest", False))),
                    collision=int(collision),
                    offroad=int(bool(latest_info.get("out_of_road", False))),
                    truncated=int(bool(truncated) or (steps >= args.max_steps and not terminated)),
                    reward=round(reward_total, 4),
                    seconds=round(time.perf_counter() - started, 2),
                )
                summaries.append(summary)
                summary_writer.writerow(summary)
                summary_file.flush()
                print(f"seed={seed} success={summary['success']} steps={steps} "
                      f"offroad={summary['offroad']} collision={summary['collision']} "
                      f"elapsed={summary['seconds']}s", flush=True)
    finally:
        if env is not None:
            env.close()

    if summaries:
        success = sum(x["success"] for x in summaries) / len(summaries)
        frames = sum(x["frames_saved"] for x in summaries)
        print(f"\nDone: {len(summaries)} routes | {frames} RGB-action pairs | "
              f"expert success {success:.1%}")
        print(f"Saved to: {run.resolve()}")
        print("IMPORTANT: This is expert demonstration collection, not a student benchmark score.")
    else:
        print("No episodes collected. Check free disk space and configuration.")


if __name__ == "__main__":
    main()
