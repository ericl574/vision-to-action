"""Collect expert labels on states visited by a camera-policy/expert mixture.

The student sees only RGB history and speed. During data collection, the
privileged expert supplies labels and partially corrects the executed steering
to keep the rollout near recoverable states. The expert is never used by the
benchmark policy at inference time.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.metadrive.action_model import ActionPolicy
from benchmarks.metadrive.collect_expert import rgb_from_observation
from benchmarks.metadrive.run import is_safe_success


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-checkpoint", type=Path,
                        default=Path("checkpoints/metadrive_actions_v1/best.pt"))
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1400)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--beta", type=float, default=0.5,
                        help="Expert share of executed steering; label is always pure expert steering")
    parser.add_argument("--traffic-density", type=float, default=0.05)
    parser.add_argument("--map-blocks", type=int, default=3)
    parser.add_argument("--quality", type=int, default=80)
    parser.add_argument("--min-free-gb", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=Path("data/collected/metadrive_expert_v1"))
    args = parser.parse_args()
    if args.episodes < 1 or args.max_steps < 1 or args.seed < 1 or args.seed + args.episodes > 5000:
        parser.error("Invalid episode/seed range; 5000+ reserved for evaluation")
    if not 0 <= args.beta <= 1 or not 0 <= args.traffic_density <= 1:
        parser.error("beta and traffic density must be in [0, 1]")
    if args.map_blocks < 1 or not 20 <= args.quality <= 100:
        parser.error("Invalid map blocks or JPEG quality")
    run = args.output / f"seed_{args.seed}_n{args.episodes}_blocks{args.map_blocks}_dagger"
    if run.exists():
        raise FileExistsError(run)
    if shutil.disk_usage(Path.cwd()).free < args.min_free_gb * 1024**3:
        raise RuntimeError("Not enough disk space for DAgger collection")

    from metadrive import MetaDriveEnv
    from metadrive.component.sensors.rgb_camera import RGBCamera
    from metadrive.examples.ppo_expert.numpy_expert import expert

    policy = ActionPolicy(args.student_checkpoint, target_speed_kmh=20.0)
    config = dict(use_render=False, image_observation=True, image_on_cuda=False,
                  norm_pixel=False, stack_size=1,
                  sensors={"rgb_camera": (RGBCamera, 192, 120)},
                  vehicle_config={"image_source": "rgb_camera"},
                  start_seed=args.seed, num_scenarios=args.episodes,
                  map=args.map_blocks, traffic_density=args.traffic_density,
                  horizon=args.max_steps)
    run.mkdir(parents=True, exist_ok=False)
    (run / "manifest.json").write_text(json.dumps(dict(
        created_utc=datetime.now(timezone.utc).isoformat(), environment="MetaDrive 0.4.3",
        collection="DAgger-style mixture rollout", student_checkpoint=str(args.student_checkpoint),
        expert="numpy PPO expert, privileged lidar/state (labels only)",
        student_inputs="four RGB frames at 0.5s spacing + speed m/s",
        expert_steering_share=args.beta, student_acceleration="20 km/h proportional servo",
        seed_start=args.seed, episodes=args.episodes, map_blocks=args.map_blocks,
        traffic_density=args.traffic_density, max_steps=args.max_steps,
        image_size=[192, 120], image_format="RGB JPEG", image_quality=args.quality,
        benchmark_seeds_reserved="5000+",
    ), indent=2))
    fields = ["seed", "step", "image", "steering", "acceleration", "speed_mps", "reward",
              "student_steering", "executed_steering", "position_x", "position_y"]
    summary_fields = ["seed", "steps", "frames_saved", "success", "collision", "offroad",
                      "truncated", "reward", "seconds"]
    env = None
    summaries = []
    try:
        env = MetaDriveEnv(config)
        with (run / "episodes.csv").open("w", newline="") as summary_file:
            summary_writer = csv.DictWriter(summary_file, fieldnames=summary_fields)
            summary_writer.writeheader()
            for episode_index in range(args.episodes):
                if shutil.disk_usage(Path.cwd()).free < args.min_free_gb * 1024**3:
                    print("Disk threshold reached; stopping cleanly", flush=True)
                    break
                seed = args.seed + episode_index
                episode_dir = run / f"episode_{seed}"
                frames_dir = episode_dir / "frames"
                frames_dir.mkdir(parents=True)
                started = time.perf_counter()
                observation, _ = env.reset(seed=seed)
                policy.reset(rgb_from_observation(observation))
                reward_total, latest_info, terminated, truncated, steps = 0.0, {}, False, False, 0
                with (episode_dir / "actions.csv").open("w", newline="") as action_file:
                    writer = csv.DictWriter(action_file, fieldnames=fields)
                    writer.writeheader()
                    for t in range(args.max_steps):
                        if t % 50 == 0 and shutil.disk_usage(Path.cwd()).free < args.min_free_gb * 1024**3:
                            raise RuntimeError(f"Disk threshold reached during seed {seed}")
                        rgb = rgb_from_observation(observation)
                        speed = float(env.agent.speed)
                        student_steer, student_accel, _ = policy.act(rgb, speed, t)
                        teacher = np.asarray(expert(env.agent, deterministic=True), dtype=np.float32).reshape(-1)
                        if teacher.shape != (2,) or not np.isfinite(teacher).all():
                            raise ValueError(f"Invalid expert action at seed={seed}, step={t}")
                        teacher = np.clip(teacher, -1.0, 1.0)
                        executed_steer = float(np.clip(args.beta * teacher[0] +
                                                       (1 - args.beta) * student_steer, -1.0, 1.0))
                        image_rel = f"frames/{t:06d}.jpg"
                        Image.fromarray(rgb).save(frames_dir / f"{t:06d}.jpg", quality=args.quality)
                        position = np.asarray(env.agent.position, dtype=np.float64)
                        observation, reward, terminated, truncated, latest_info = env.step(
                            [executed_steer, student_accel])
                        reward_total += float(reward)
                        writer.writerow(dict(seed=seed, step=t, image=image_rel,
                                             steering=f"{teacher[0]:.7f}", acceleration=f"{teacher[1]:.7f}",
                                             speed_mps=f"{speed:.5f}", reward=f"{float(reward):.5f}",
                                             student_steering=f"{student_steer:.7f}",
                                             executed_steering=f"{executed_steer:.7f}",
                                             position_x=f"{position[0]:.6f}", position_y=f"{position[1]:.6f}"))
                        steps += 1
                        if terminated or truncated:
                            break
                collision = any(latest_info.get(key, False) for key in
                                ("crash", "crash_vehicle", "crash_object", "crash_building", "crash_human"))
                summary = dict(seed=seed, steps=steps, frames_saved=steps,
                               success=int(is_safe_success(latest_info)),
                               collision=int(collision), offroad=int(bool(latest_info.get("out_of_road", False))),
                               truncated=int(bool(truncated) or (steps >= args.max_steps and not terminated)),
                               reward=round(reward_total, 4), seconds=round(time.perf_counter() - started, 2))
                summaries.append(summary)
                summary_writer.writerow(summary)
                summary_file.flush()
                print(f"seed={seed} success={summary['success']} steps={steps} "
                      f"offroad={summary['offroad']} collision={summary['collision']}", flush=True)
    finally:
        if env is not None:
            env.close()
    print(f"DAgger collection: {len(summaries)} routes, {sum(x['frames_saved'] for x in summaries)} "
          f"expert-labeled camera states, success={sum(x['success'] for x in summaries) / max(1,len(summaries)):.1%}",
          flush=True)


if __name__ == "__main__":
    main()
