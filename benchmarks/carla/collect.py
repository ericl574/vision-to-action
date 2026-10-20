"""Collect expert demonstrations or DAgger data in CARLA.

    # expert demonstrations (teacher drives, with recovery perturbations)
    python benchmarks/carla/collect.py --mode expert --routes-file benchmarks/carla/routes/train_town10.json \
        --out /workspace/data/expert_town10 --port 2000 --tm-port 8000 --launch

    # DAgger: student drives (mixed with the expert by --beta), expert labels every visited state
    python benchmarks/carla/collect.py --mode dagger --student-checkpoint /workspace/runs/pilot_v1/best.pt --beta 0.5 \
        --routes-file benchmarks/carla/routes/train_town10.json --exclude-split /workspace/runs/pilot_v1/split.json \
        --out /workspace/data/dagger1_town10 --tag d1 --port 2000 --tm-port 8000 --launch

Finished routes are skipped on rerun, so a killed collector can simply be restarted.
"""

import argparse
import json
import random
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.carla.record import RouteRecorder, load_manifest  # noqa: E402
from benchmarks.carla.routes import load_routes  # noqa: E402
from benchmarks.carla.server import kill_server, launch_server, wait_for_server  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("expert", "dagger"), default="expert")
    parser.add_argument("--routes-file", required=True)
    parser.add_argument("--route-start", type=int, default=0)
    parser.add_argument("--route-count", type=int, default=0, help="0 = all remaining")
    parser.add_argument("--out", required=True)
    parser.add_argument("--tag", default="", help="prefix for route ids (keeps DAgger routes distinct)")
    parser.add_argument("--exclude-split", default=None, help="split.json whose val routes are skipped")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--carla-root", default=None)
    parser.add_argument("--launch", action="store_true", help="start a CARLA server on --port")
    parser.add_argument("--quality", default="Epic")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--beta", type=float, default=0.5, help="dagger: probability of executing the expert")
    parser.add_argument("--student-checkpoint", default=None)
    parser.add_argument("--vehicles", type=int, default=40)
    parser.add_argument("--walkers", type=int, default=40)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--fov", type=float, default=100.0)
    parser.add_argument("--no-perturb", action="store_true")
    parser.add_argument("--behavior", default="normal")
    parser.add_argument("--seed-offset", type=int, default=0, help="added to each route seed (fresh traffic draws)")
    return parser.parse_args()


def main():
    args = parse_args()
    from benchmarks.carla.carla_env import CarlaEnv
    from benchmarks.carla.episode import run_episode
    from benchmarks.carla.expert import Expert

    routes = load_routes(args.routes_file)
    end = len(routes) if args.route_count <= 0 else min(len(routes), args.route_start + args.route_count)
    routes = routes[args.route_start:end]
    if args.exclude_split:
        with open(args.exclude_split) as handle:
            excluded = set(json.load(handle)["val"])
        routes = [r for r in routes if r.route_id not in excluded]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    done = {m["route_id"] for m in load_manifest(out)}

    student = None
    if args.mode == "dagger":
        if not args.student_checkpoint:
            raise SystemExit("--mode dagger needs --student-checkpoint")
        from benchmarks.carla.model import PilotPolicy

        student = PilotPolicy(args.student_checkpoint)
        print(f"student {student.checkpoint} epoch {student.epoch} input {student.width}x{student.height}")

    proc = launch_server(args.carla_root, args.port, args.quality, log_path=str(out / "carla_server.log")) if args.launch else None
    env = None
    try:
        client = wait_for_server(port=args.port)
        env = CarlaEnv(client, args.width, args.height, args.fov, vehicles=args.vehicles, walkers=args.walkers, tm_port=args.tm_port)
        total_frames = 0
        for i, spec in enumerate(routes):
            route_id = f"{args.tag}_{spec.route_id}" if args.tag else spec.route_id
            if route_id in done:
                continue
            spec.seed = spec.seed + args.seed_offset
            rng = random.Random(spec.seed + 17)
            recorder = RouteRecorder(out, route_id, {"town": spec.town, "weather": spec.weather, "mode": args.mode,
                                                     "beta": args.beta if args.mode == "dagger" else 1.0,
                                                     "student": args.student_checkpoint or "", "source_route_id": spec.route_id,
                                                     "seed": spec.seed, "width": args.width, "height": args.height, "fov": args.fov})
            state = {"expert": None}

            def act_fn(obs, info, step, state=state, rng=rng, recorder=recorder):
                if state["expert"] is None:
                    state["expert"] = Expert(env, args.behavior, perturb=not args.no_perturb and args.mode == "expert",
                                             seed=spec.seed, dt=env.fixed_dt)
                    if student is not None:
                        student.reset(obs["rgb"])
                expert = state["expert"]
                label = expert.act(info)
                row = {
                    "t": info["t"], "speed_mps": obs["speed_mps"], "command": obs["command"],
                    "steer": label["steer"], "throttle": label["throttle"], "brake": label["brake"],
                    "target_speed_mps": label["target_speed_mps"], "emergency": label["emergency"],
                    "tl_state": info["tl_state"], "tl_dist_m": round(info["tl_dist_m"], 2),
                    "hazard": info["hazard"], "hazard_dist_m": round(info["hazard_dist_m"], 2), "hazard_type": info["hazard_type"],
                    "route_completion": round(info["route_completion"], 4), "x": round(info["x"], 2), "y": round(info["y"], 2),
                    "yaw": round(info["yaw"], 2),
                }
                if student is not None:
                    s_steer, s_throttle, s_brake, outputs = student.act(obs["rgb"], obs["speed_mps"], obs["command"])
                    use_expert = rng.random() < args.beta
                    row.update({"student_steer": round(s_steer, 4), "student_target_speed_mps": round(outputs["target_speed_mps"], 3),
                                "used_expert": int(use_expert), "is_perturbed": 0})
                    if use_expert:
                        exec_control = (label["steer"], label["throttle"], label["brake"])
                    else:
                        exec_control = (s_steer, s_throttle, s_brake)
                else:
                    offset, perturbed = expert.perturbation(obs["command"], label["steer"])
                    exec_control = (max(-1.0, min(1.0, label["steer"] + offset)), label["throttle"], label["brake"])
                    row.update({"is_perturbed": int(perturbed), "used_expert": 1})
                row.update({"exec_steer": round(exec_control[0], 4), "exec_throttle": round(exec_control[1], 4),
                            "exec_brake": round(exec_control[2], 4)})
                recorder.append(obs["rgb"], row)
                return exec_control[0], exec_control[1], exec_control[2], None

            try:
                result = run_episode(env, spec, act_fn, args.max_steps)
            except Exception:  # noqa: BLE001 - keep collecting; the route stays unfinished and is retried on rerun
                traceback.print_exc()
                print(f"[{i + 1}/{len(routes)}] {route_id}: ERROR, reconnecting", flush=True)
                try:
                    env.cleanup()
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(5)
                client = wait_for_server(port=args.port)
                env = CarlaEnv(client, args.width, args.height, args.fov, vehicles=args.vehicles, walkers=args.walkers, tm_port=args.tm_port)
                continue
            recorder.finish(result["status"], result)
            total_frames += len(recorder.rows)
            print(f"[{i + 1}/{len(routes)}] {route_id}: {result['status']} completion={result['route_completion']:.2f} "
                  f"frames={len(recorder.rows)} red={result['red_light_violations']} total_frames={total_frames} "
                  f"({result['wall_seconds']:.0f}s)", flush=True)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:  # noqa: BLE001
                pass
        kill_server(proc)


if __name__ == "__main__":
    main()
