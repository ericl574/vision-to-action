"""Closed-loop evaluation on a route split, with per-step logs for the overlay.

    python benchmarks/carla/run.py --policy pilot --checkpoint /workspace/runs/pilot_v1/best.pt \
        --routes-file benchmarks/carla/routes/dev_town10.json \
        --out benchmarks/carla/results/pilot_v1_dev_town10.csv --log-dir /workspace/runs/eval_pilot_v1_dev_town10 --launch

    python benchmarks/carla/run.py --policy expert --routes-file benchmarks/carla/routes/dev_town10.json \
        --out benchmarks/carla/results/expert_dev_town10.csv --log-dir /workspace/runs/eval_expert_dev_town10 --launch

The student policy receives only the camera frame, its own speed and the navigation command.
"""

import argparse
import csv
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.carla.metrics import summarize, write_results  # noqa: E402
from benchmarks.carla.record import LABEL_COLUMNS, OUTPUT_COLUMNS, RouteRecorder  # noqa: E402
from benchmarks.carla.routes import load_routes  # noqa: E402
from benchmarks.carla.server import kill_server, launch_server, wait_for_server  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=("pilot", "expert"), required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--routes-file", required=True)
    parser.add_argument("--route-start", type=int, default=0)
    parser.add_argument("--route-count", type=int, default=0)
    parser.add_argument("--out", required=True, help="results CSV (a JSON twin is written next to it)")
    parser.add_argument("--log-dir", default=None, help="write frames + per-step outputs here for the overlay")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--carla-root", default=None)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--quality", default="Epic")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--vehicles", type=int, default=40)
    parser.add_argument("--walkers", type=int, default=40)
    parser.add_argument("--width", type=int, default=1024, help="capture size; the policy resizes to its own input")
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--fov", type=float, default=100.0)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    from benchmarks.carla.carla_env import CarlaEnv
    from benchmarks.carla.episode import run_episode

    routes = load_routes(args.routes_file)
    end = len(routes) if args.route_count <= 0 else min(len(routes), args.route_start + args.route_count)
    routes = routes[args.route_start:end]
    out_csv = Path(args.out)
    rows = []
    if out_csv.exists():
        with open(out_csv, newline="") as handle:
            rows = list(csv.DictReader(handle))
        print(f"resuming: {len(rows)} routes already in {out_csv}")
    done = {row["route_id"] for row in rows}

    policy = None
    if args.policy == "pilot":
        if not args.checkpoint:
            raise SystemExit("--policy pilot needs --checkpoint")
        from benchmarks.carla.model import PilotPolicy

        policy = PilotPolicy(args.checkpoint, device=args.device)
        print(f"loaded {policy.checkpoint} epoch={policy.epoch} input={policy.width}x{policy.height} device={policy.device}")

    proc = launch_server(args.carla_root, args.port, args.quality, log_path=str(out_csv.with_suffix(".server.log"))) if args.launch else None
    env = None
    meta = {"policy": args.policy, "checkpoint": args.checkpoint or "", "routes_file": args.routes_file,
            "max_steps": args.max_steps, "vehicles": args.vehicles, "walkers": args.walkers, "fov": args.fov}
    try:
        client = wait_for_server(port=args.port)
        env = CarlaEnv(client, args.width, args.height, args.fov, vehicles=args.vehicles, walkers=args.walkers, tm_port=args.tm_port)
        for i, spec in enumerate(routes):
            if spec.route_id in done:
                continue
            spec.seed = spec.seed + args.seed_offset
            recorder = None
            if args.log_dir:
                columns = OUTPUT_COLUMNS if policy is not None else LABEL_COLUMNS
                recorder = RouteRecorder(args.log_dir, spec.route_id, {"town": spec.town, "weather": spec.weather, "policy": args.policy},
                                         jpeg_quality=85, columns=columns, csv_name="outputs.csv" if policy is not None else "labels.csv")
            state = {"expert": None}

            def act_fn(obs, info, step, state=state, recorder=recorder):
                if policy is not None:
                    if step == 0:
                        policy.reset(obs["rgb"])
                    steer, throttle, brake, outputs = policy.act(obs["rgb"], obs["speed_mps"], obs["command"])
                    if recorder is not None:
                        tl = outputs["tl_probs"]
                        recorder.append(obs["rgb"], {
                            "t": info["t"], "speed_mps": obs["speed_mps"], "command": obs["command"], "steer": round(steer, 4),
                            "throttle": round(throttle, 4), "brake": round(brake, 4), "target_speed_mps": round(outputs["target_speed_mps"], 3),
                            "tl_none": round(tl[0], 3), "tl_green": round(tl[1], 3), "tl_yellow": round(tl[2], 3), "tl_red": round(tl[3], 3),
                            "hazard_prob": round(outputs["hazard_prob"], 3), "hazard_dist_m": round(outputs["hazard_dist_m"], 2),
                            "gt_tl_state": info["tl_state"], "gt_hazard": info["hazard"], "route_completion": round(info["route_completion"], 4),
                            "ran_red_light": int(info["ran_red_light"]), "x": round(info["x"], 2), "y": round(info["y"], 2)})
                    return steer, throttle, brake, None
                from benchmarks.carla.expert import Expert

                if state["expert"] is None:
                    state["expert"] = Expert(env, perturb=False, dt=env.fixed_dt)
                label = state["expert"].act(info)
                if recorder is not None:
                    recorder.append(obs["rgb"], {
                        "t": info["t"], "speed_mps": obs["speed_mps"], "command": obs["command"], "steer": label["steer"],
                        "throttle": label["throttle"], "brake": label["brake"], "target_speed_mps": label["target_speed_mps"],
                        "emergency": label["emergency"], "tl_state": info["tl_state"], "tl_dist_m": round(info["tl_dist_m"], 2),
                        "hazard": info["hazard"], "hazard_dist_m": round(info["hazard_dist_m"], 2), "hazard_type": info["hazard_type"],
                        "route_completion": round(info["route_completion"], 4), "x": round(info["x"], 2), "y": round(info["y"], 2),
                        "yaw": round(info["yaw"], 2)})
                return label["steer"], label["throttle"], label["brake"], None

            try:
                result = run_episode(env, spec, act_fn, args.max_steps)
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                print(f"[{i + 1}/{len(routes)}] {spec.route_id}: ERROR, reconnecting", flush=True)
                try:
                    env.cleanup()
                except Exception:  # noqa: BLE001
                    pass
                client = wait_for_server(port=args.port)
                env = CarlaEnv(client, args.width, args.height, args.fov, vehicles=args.vehicles, walkers=args.walkers, tm_port=args.tm_port)
                continue
            if recorder is not None:
                recorder.finish(result["status"], result)
            rows.append(result)
            write_results(rows, out_csv, meta)
            print(f"[{i + 1}/{len(routes)}] {spec.route_id}: {result['status']} completion={result['route_completion']:.2f} "
                  f"success={result['success']} red={result['red_light_violations']} ({result['wall_seconds']:.0f}s)", flush=True)
        summary = summarize(rows)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary.items()})
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:  # noqa: BLE001
                pass
        kill_server(proc)


if __name__ == "__main__":
    main()
