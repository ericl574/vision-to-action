"""Generate seeded route files (committed under benchmarks/carla/routes/).

    python benchmarks/carla/make_routes.py --town Town10HD_Opt --n 60 --seed 100 \
        --out benchmarks/carla/routes/train_town10.json --launch
    python benchmarks/carla/make_routes.py --town Town10HD_Opt --n 10 --seed 500 \
        --out benchmarks/carla/routes/dev_town10.json --exclude benchmarks/carla/routes/train_town10.json --launch

Routes are spawn-point pairs whose planned path is 300-800 m, contains at least one junction
manoeuvre and no lane change. ``--exclude`` keeps held-out files disjoint from training pairs.
"""

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.carla.routes import RouteSpec, load_routes, save_routes  # noqa: E402
from benchmarks.carla.server import kill_server, launch_server, wait_for_server  # noqa: E402

WEATHERS = ("ClearNoon", "CloudyNoon", "WetNoon", "SoftRainSunset")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--town", required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prefix", default=None, help="route id prefix (default: derived from the file name)")
    parser.add_argument("--min-m", type=float, default=300.0)
    parser.add_argument("--max-m", type=float, default=800.0)
    parser.add_argument("--min-turns", type=int, default=1)
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--carla-root", default=None)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()

    from benchmarks.carla.carla_env import CarlaEnv

    excluded = set()
    for path in args.exclude:
        excluded.update((r.town, r.start_idx, r.end_idx) for r in load_routes(path))
    prefix = args.prefix or Path(args.out).stem
    proc = launch_server(args.carla_root, args.port) if args.launch else None
    try:
        client = wait_for_server(port=args.port)
        env = CarlaEnv(client)
        env.load_town(args.town)
        n_points = len(env.spawn_points)
        rng = random.Random(args.seed)
        routes, seen, attempts = [], set(), 0
        while len(routes) < args.n and attempts < args.n * 200:
            attempts += 1
            start, end = rng.randrange(n_points), rng.randrange(n_points)
            if start == end or (args.town, start, end) in seen or (args.town, start, end) in excluded:
                continue
            seen.add((args.town, start, end))
            plan = env.trace(start, end)
            length = env.plan_length_m(plan)
            if not (args.min_m <= length <= args.max_m) or env.plan_has_lane_change(plan) or env.plan_turns(plan) < args.min_turns:
                continue
            routes.append(RouteSpec(args.town, f"{prefix}_{len(routes):03d}", start, end,
                                    WEATHERS[len(routes) % len(WEATHERS)], round(length, 1), args.seed * 1000 + len(routes)))
        save_routes(args.out, routes)
        lengths = [r.length_m for r in routes]
        print(f"wrote {len(routes)} routes to {args.out} (mean {sum(lengths) / max(1, len(lengths)):.0f} m, "
              f"{attempts} attempts, {n_points} spawn points)")
        env.close()
    finally:
        kill_server(proc)


if __name__ == "__main__":
    main()
