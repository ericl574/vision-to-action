"""One closed-loop episode on one route. Shared by collection and evaluation."""

import math
import time

from benchmarks.carla.metrics import check_termination, route_outcome

STUCK_LIMIT_STEPS = 300  # 30 s without moving, with no red light or hazard to wait for


def run_episode(env, spec, act_fn, max_steps: int, on_step=None) -> dict:
    """``act_fn(obs, info, step) -> (steer, throttle, brake, extra)``; ``on_step(obs, info, step, control, extra)``
    sees the state *before* the control is applied, which is the state the labels describe."""
    obs, info = env.reset(spec)
    off_road_steps = deviation_steps = stuck_steps = 0
    distance_m = 0.0
    last_xy = (info["x"], info["y"])
    status = None
    started = time.time()
    step = 0
    for step in range(max_steps):
        steer, throttle, brake, extra = act_fn(obs, info, step)
        if on_step is not None:
            on_step(obs, info, step, (steer, throttle, brake), extra)
        obs, info = env.step(steer, throttle, brake)
        distance_m += math.hypot(info["x"] - last_xy[0], info["y"] - last_xy[1])
        last_xy = (info["x"], info["y"])
        off_road_steps = off_road_steps + 1 if info["off_road"] else 0
        deviation_steps = deviation_steps + 1 if info["deviated"] else 0
        waiting = info["tl_state"] == 3 or info["hazard"]
        stuck_steps = stuck_steps + 1 if (info["speed_mps"] < 0.1 and not waiting) else 0
        status = check_termination(step, max_steps, info["collision_type"], off_road_steps, deviation_steps,
                                   info["arrived"], stuck_steps, stuck_limit=STUCK_LIMIT_STEPS)
        if status:
            break
    result = route_outcome(status or "timeout", info["route_completion"], info["red_light_violations"], info["collision_type"])
    result.update({
        "route_id": spec.route_id, "town": spec.town, "weather": spec.weather, "route_length_m": round(env.tracker.length_m, 1),
        "distance_m": round(distance_m, 1), "steps": step + 1, "sim_seconds": round((step + 1) * env.fixed_dt, 1),
        "wall_seconds": round(time.time() - started, 1), "collision_other": info["collision_other"],
    })
    return result
