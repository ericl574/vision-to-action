"""Closed-loop MetaDrive benchmark for camera-only driving policies.

Examples from the repository root::

    python benchmarks/metadrive/run.py --policy action --episodes 5
    python benchmarks/metadrive/run.py --policy zero --episodes 5
    python benchmarks/metadrive/run.py --policy action --episodes 1 --max-steps 300 \
        --record benchmarks/metadrive/model_demo.mp4

Both learned policies use only RGB camera frames and ego speed. Neither calls
MetaDrive's expert or privileged lane observations. The zero policy is a
simple baseline with the same longitudinal speed servo.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_controller import ModelTrajectoryPolicy, ZeroSteeringPolicy
from benchmarks.metadrive.action_model import ActionPolicy


def camera_rgb(env, width: int = 192, height: int = 120) -> np.ndarray:
    """Read MetaDrive's BGR camera and convert it to training-order RGB."""
    bgr = env.engine.get_sensor("rgb_camera").perceive(to_float=False)
    image = np.asarray(bgr)
    if image.ndim == 4:
        image = image[..., -1]
    if image.shape != (height, width, 3) or image.dtype != np.uint8:
        raise ValueError(f"Expected MetaDrive BGR uint8 image {(height, width, 3)}, got {image.shape} {image.dtype}")
    return np.ascontiguousarray(image[..., ::-1])


def policy_rgb(rgb: np.ndarray) -> np.ndarray:
    """Keep the trained model's 192x120 input across recording resolutions."""
    if rgb.shape[:2] == (120, 192):
        return rgb
    import cv2

    return np.ascontiguousarray(cv2.resize(rgb, (192, 120), interpolation=cv2.INTER_AREA))


def build_policy(args):
    common = dict(target_speed_kmh=args.target_kmh)
    if args.policy == "action":
        policy = ActionPolicy(args.checkpoint, device=args.device,
                              history_interval=args.history_interval, **common)
        print(f"Loaded direct-action model: {policy.checkpoint} | device={policy.device} | "
              f"epoch={policy.epoch} | val_MAE={policy.validation_mae}")
        return policy
    if args.policy in ("model", "trajectory"):
        policy = ModelTrajectoryPolicy(
            args.checkpoint,
            device=args.device,
            history_interval=args.history_interval,
            steering_gain=args.steering_gain,
            lookahead_m=args.lookahead_m,
            **common,
        )
        print(
            f"Loaded trajectory model: {policy.model.checkpoint} | "
            f"device={policy.model.device} | epoch={policy.model.epoch} | "
            f"val_ADE={policy.model.validation_ade}"
        )
        return policy
    if args.policy == "zero":
        print("Using baseline: zero steering + proportional speed control")
        return ZeroSteeringPolicy(**common)
    raise ValueError(f"Unknown policy: {args.policy}")


def collision_from_info(info: dict) -> bool:
    return bool(any(info.get(key, False) for key in (
        "crash", "crash_vehicle", "crash_object", "crash_building", "crash_human"
    )))


def is_safe_success(info: dict) -> bool:
    """Reject MetaDrive arrival flags on collision/offroad or at map spawn."""
    return (bool(info.get("arrive_dest", False)) and not collision_from_info(info)
            and not bool(info.get("out_of_road", False))
            and float(info.get("route_completion", 0.0)) >= 0.95)


def make_video_writer(path: Path, rgb: np.ndarray):
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (1056, 540),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    return writer


def write_frame(writer, rgb: np.ndarray, policy_name: str, seed: int, step: int,
                steering: float, acceleration: float, speed_kmh: float, trajectory=None):
    import cv2

    camera = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    interpolation = cv2.INTER_AREA if rgb.shape[1] > 768 else cv2.INTER_LINEAR
    camera = cv2.resize(camera, (768, 480), interpolation=interpolation)
    height, width = camera.shape[:2]
    frame = np.full((height + 60, width + 288, 3), (28, 24, 20), dtype=np.uint8)
    frame[60:, :width] = camera
    model_policy = policy_name in ("model", "trajectory", "action")
    title = ("CAMERA-ONLY ACTION MODEL" if policy_name == "action" else
             "CAMERA-ONLY TRAJECTORY MODEL" if model_policy else "ZERO-STEERING BASELINE")
    cv2.putText(frame, title, (16, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 220, 255), 2, cv2.LINE_AA)
    subtitle = "RGB + speed  ->  direct steering" if policy_name == "action" else "RGB + speed  ->  5s path  ->  steering"
    cv2.putText(frame, subtitle, (16, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
    panel_x = width + 16
    cv2.putText(frame, f"Road seed {seed}", (panel_x, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Time {step/10:.1f}s", (panel_x, 93), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Speed {speed_kmh:.1f} km/h", (panel_x, 121), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Steer {steering:+.2f}", (panel_x, 149), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Accel {acceleration:+.2f}", (panel_x, 177), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, "Predicted path" if policy_name != "action" else "Model steering",
                (panel_x, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (80, 220, 255), 1, cv2.LINE_AA)
    origin = (width + 144, height + 30)
    cv2.line(frame, (origin[0], 260), origin, (100, 100, 100), 1)
    cv2.circle(frame, origin, 5, (255, 255, 255), -1)
    if policy_name == "action":
        gauge_x = int(np.clip(origin[0] + steering * 110, width + 16, width + 272))
        cv2.line(frame, (width + 32, 310), (width + 256, 310), (110, 110, 110), 2)
        cv2.circle(frame, (gauge_x, 310), 8, (80, 220, 255), -1)
    if trajectory is not None:
        points = np.asarray(trajectory)
        if points.shape == (20, 2) and np.isfinite(points).all():
            path = np.array([(int(origin[0] + 10 * y), int(origin[1] - 10 * x))
                             for x, y in points], dtype=np.int32)
            path[:, 0] = np.clip(path[:, 0], width + 8, width + 280)
            path[:, 1] = np.clip(path[:, 1], 250, height + 54)
            cv2.polylines(frame, [np.vstack((np.asarray(origin), path)).reshape(-1, 1, 2)],
                          False, (80, 220, 255), 2, cv2.LINE_AA)
    writer.write(frame)


def evaluate_episode(env, policy, args, seed: int, writer=None) -> dict:
    env.reset(seed=seed)
    first_rgb = camera_rgb(env, args.camera_width, args.camera_height)
    policy.reset(policy_rgb(first_rgb))
    previous_position = np.asarray(env.agent.position, dtype=np.float64).copy()
    distance_traveled = 0.0
    info = {}
    diagnostics = {}
    speeds = []
    absolute_steering = []
    done = False
    terminated = False
    truncated = False
    steps = 0

    while steps < args.max_steps and not done:
        rgb = camera_rgb(env, args.camera_width, args.camera_height)
        speed_mps = float(env.agent.speed)
        steering, acceleration, diagnostics = policy.act(policy_rgb(rgb), speed_mps, steps)
        if not np.isfinite([steering, acceleration]).all():
            raise ValueError(f"Non-finite action at seed={seed}, step={steps}: {steering}, {acceleration}")
        steering = float(np.clip(steering, -1.0, 1.0))
        acceleration = float(np.clip(acceleration, -1.0, 1.0))
        if writer is not None:
            write_frame(writer, rgb, args.policy, seed, steps, steering, acceleration,
                        speed_mps * 3.6, getattr(policy, "latest_trajectory", None))

        _obs, _reward, terminated, truncated, info = env.step([steering, acceleration])
        position = np.asarray(env.agent.position, dtype=np.float64)
        distance_traveled += float(np.linalg.norm(position - previous_position))
        previous_position = position
        speeds.append(speed_mps * 3.6)
        absolute_steering.append(abs(steering))
        steps += 1
        done = bool(terminated or truncated)

    collision = collision_from_info(info)
    offroad = bool(info.get("out_of_road", False))
    route_completion = float(info.get("route_completion", np.nan))
    raw_arrival = bool(info.get("arrive_dest", False))
    safe_success = is_safe_success(info)
    row = {
        "policy": args.policy,
        "seed": seed,
        "steps": steps,
        "success": int(safe_success),
        "arrive_dest_raw": int(raw_arrival),
        "collision": int(collision),
        "offroad": int(offroad),
        "route_completion": route_completion,
        "distance_traveled_m": distance_traveled,
        "mean_speed_kmh": float(np.mean(speeds)) if speeds else 0.0,
        "mean_abs_steering": float(np.mean(absolute_steering)) if absolute_steering else 0.0,
        "inference_count": int(getattr(policy, "inference_count", 0)),
        "last_inference_step": diagnostics.get("inference_step"),
        "terminated": int(terminated),
        "truncated": int(truncated),
        "max_steps_reached": int(truncated or (steps >= args.max_steps and not done)),
    }
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=["model", "trajectory", "action", "zero"], default="action")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "benchmarks/metadrive/weights/action_v2_dagger_best.pt")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=5000)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--target-kmh", type=float, default=20.0)
    parser.add_argument("--history-interval", type=int, default=5,
                        help="MetaDrive steps between history frames; default matches 0.5s training spacing")
    parser.add_argument("--lookahead-m", type=float, default=4.0)
    parser.add_argument("--steering-gain", type=float, default=1.0)
    parser.add_argument("--traffic-density", type=float, default=0.05)
    parser.add_argument("--map-blocks", type=int, default=3)
    parser.add_argument("--camera-width", type=int, default=192)
    parser.add_argument("--camera-height", type=int, default=120)
    parser.add_argument("--render", action="store_true", help="Open MetaDrive's onscreen renderer")
    parser.add_argument("--record", type=Path, default=None, help="Write an annotated MP4 of all episodes")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/metadrive/results_model.csv")
    args = parser.parse_args()
    if args.episodes < 1 or args.max_steps < 1 or args.history_interval < 1:
        parser.error("episodes, max-steps and history-interval must be positive")
    if args.target_kmh <= 0 or args.lookahead_m <= 0 or args.steering_gain < 0:
        parser.error("target-kmh and lookahead-m must be positive; steering-gain must be nonnegative")
    if not 0 <= args.traffic_density <= 1 or args.map_blocks < 1:
        parser.error("traffic-density must be between 0 and 1 and map-blocks must be positive")
    if args.camera_width < 192 or args.camera_height < 120 or args.camera_width * 5 != args.camera_height * 8:
        parser.error("camera must be at least 192x120 and keep the training 8:5 aspect ratio")

    policy = build_policy(args)
    from metadrive import MetaDriveEnv
    from metadrive.component.sensors.rgb_camera import RGBCamera

    env = MetaDriveEnv(dict(
        use_render=args.render,
        image_observation=True,
        image_on_cuda=False,
        norm_pixel=False,
        stack_size=1,
        sensors={"rgb_camera": (RGBCamera, args.camera_width, args.camera_height)},
        vehicle_config={"image_source": "rgb_camera"},
        num_scenarios=args.episodes,
        start_seed=args.seed,
        traffic_density=args.traffic_density,
        map=args.map_blocks,
        horizon=args.max_steps,
    ))
    video_writer = None
    rows = []
    try:
        for index in range(args.episodes):
            seed = args.seed + index
            if args.record is not None and video_writer is None:
                env.reset(seed=seed)
                video_writer = make_video_writer(args.record,
                                                 camera_rgb(env, args.camera_width, args.camera_height))
            row = evaluate_episode(env, policy, args, seed, writer=video_writer)
            rows.append(row)
            print(
                f"seed={seed}: steps={row['steps']} success={row['success']} "
                f"collision={row['collision']} offroad={row['offroad']} "
                f"route={row['route_completion']:.3f} distance={row['distance_traveled_m']:.1f}m"
            )
    except Exception as error:
        if "Could not open window" in str(error):
            raise RuntimeError(
                "MetaDrive camera rendering could not open a Panda3D window. "
                "Run this command from a logged-in Mac desktop session; the "
                "physics-only environment can run headlessly, but RGB camera "
                "benchmarks require a display in MetaDrive 0.4.3."
            ) from error
        raise
    finally:
        if video_writer is not None:
            video_writer.release()
        env.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with args.output.open("w", newline="") as output:
        writer_csv = csv.DictWriter(output, fieldnames=fields)
        writer_csv.writeheader()
        writer_csv.writerows(rows)
    summary = {
        "policy": args.policy,
        "checkpoint": str(args.checkpoint) if args.policy in ("model", "trajectory", "action") else None,
        "seed_start": args.seed,
        "episodes": len(rows),
        "config": {**vars(args), "checkpoint": str(args.checkpoint), "output": str(args.output),
                   "record": str(args.record) if args.record else None},
        "metrics": {
            "success_rate": float(np.mean([r["success"] for r in rows])),
            "collision_rate": float(np.mean([r["collision"] for r in rows])),
            "offroad_rate": float(np.mean([r["offroad"] for r in rows])),
            "mean_route_completion": float(np.nanmean([r["route_completion"] for r in rows])),
            "mean_distance_traveled_m": float(np.mean([r["distance_traveled_m"] for r in rows])),
        },
        "episodes_detail": rows,
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"Success rate: {summary['metrics']['success_rate']:.1%}")
    print(f"Collision rate: {summary['metrics']['collision_rate']:.1%} | Off-road rate: {summary['metrics']['offroad_rate']:.1%}")
    print(f"Mean route completion: {summary['metrics']['mean_route_completion']:.3f}")
    print(f"Metrics saved to: {args.output} and {args.output.with_suffix('.json')}")
    if args.record:
        print(f"Video saved to: {args.record}")


if __name__ == "__main__":
    main()
