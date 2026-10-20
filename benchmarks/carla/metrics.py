"""Episode termination, red-light monitoring, and the success rule. Pure Python."""

import csv
import json
from pathlib import Path

SUCCESS_MIN_COMPLETION = 0.95
STATUSES = ("arrived", "collision", "off_road", "route_deviation", "timeout", "stuck")


class RedLightMonitor:
    """Edge-triggered red-light violation detector.

    Feed it, every control step, whether the ego is inside the trigger volume of the light that
    governs *its own lane* (``vehicle.is_at_traffic_light()``), that light's id and whether it is
    red, plus ego speed. Leaving the trigger volume while the light was red and still moving
    means the stop line was crossed on red. Lights for other lanes never reach this monitor.
    """

    def __init__(self, min_speed_mps: float = 0.5):
        self.min_speed_mps = min_speed_mps
        self._inside = None  # (light_id, was_red_on_last_tick)
        self.violations = 0
        self.violated_ids: list = []

    def update(self, at_light: bool, light_id, is_red: bool, speed_mps: float) -> bool:
        violation = False
        if at_light:
            self._inside = (light_id, bool(is_red))
        elif self._inside is not None:
            left_id, was_red = self._inside
            self._inside = None
            if was_red and speed_mps > self.min_speed_mps:
                violation = True
                self.violations += 1
                self.violated_ids.append(left_id)
        return violation


def check_termination(step: int, max_steps: int, collision_type, off_road_steps: int, deviation_steps: int,
                      arrived: bool, stuck_steps: int, off_road_limit: int = 10, deviation_limit: int = 10,
                      stuck_limit: int = 600):
    """Return the terminal status for this step, or ``None`` to keep driving.

    Order matters: a collision on the arrival step is still a collision (MetaDrive's spawn
    anomaly taught us not to trust arrival alone).
    """
    if collision_type:
        return "collision"
    if off_road_steps >= off_road_limit:
        return "off_road"
    if deviation_steps >= deviation_limit:
        return "route_deviation"
    if arrived:
        return "arrived"
    if stuck_steps >= stuck_limit:
        return "stuck"
    if step + 1 >= max_steps:
        return "timeout"
    return None


def route_outcome(status: str, completion: float, red_light_violations: int, collision_type=None) -> dict:
    """Apply the strict success rule to one finished route."""
    success = (
        status == "arrived"
        and completion >= SUCCESS_MIN_COMPLETION
        and not collision_type
        and red_light_violations == 0
    )
    return {
        "status": status,
        "route_completion": float(completion),
        "collision_type": collision_type or "",
        "red_light_violations": int(red_light_violations),
        "success": bool(success),
    }


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"routes": 0}

    def rate(predicate):
        return sum(1 for row in rows if predicate(row)) / n

    def as_bool(value):
        return str(value).strip().lower() in ("1", "true", "yes")

    distance_km = sum(float(row.get("distance_m", 0) or 0) for row in rows) / 1000.0
    violations = sum(int(float(row.get("red_light_violations", 0) or 0)) for row in rows)
    return {
        "routes": n,
        "successes": sum(1 for row in rows if as_bool(row.get("success"))),
        "success_rate": rate(lambda r: as_bool(r.get("success"))),
        "mean_completion": sum(float(row["route_completion"]) for row in rows) / n,
        "collision_rate": rate(lambda r: r.get("status") == "collision"),
        "collision_vehicle_rate": rate(lambda r: r.get("collision_type") == "vehicle"),
        "collision_walker_rate": rate(lambda r: r.get("collision_type") == "walker"),
        "collision_static_rate": rate(lambda r: r.get("collision_type") == "static"),
        "off_road_rate": rate(lambda r: r.get("status") == "off_road"),
        "deviation_rate": rate(lambda r: r.get("status") == "route_deviation"),
        "timeout_rate": rate(lambda r: r.get("status") in ("timeout", "stuck")),
        "red_light_violations": violations,
        "red_light_violations_per_km": (violations / distance_km) if distance_km > 0 else 0.0,
        "distance_km": distance_km,
    }


def write_results(rows: list[dict], csv_path, meta: dict | None = None) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    with open(csv_path.with_suffix(".json"), "w") as handle:
        json.dump({"meta": meta or {}, "summary": summarize(rows), "routes": rows}, handle, indent=1)


def markdown_table(named_summaries: dict[str, dict]) -> str:
    header = ("| Policy | Success | Mean completion | Collisions | Red-light violations | Off-road / deviation | Timeout |\n"
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    lines = [header]
    for name, s in named_summaries.items():
        if s.get("routes", 0) == 0:
            continue
        lines.append(
            f"| {name} | {s['successes']}/{s['routes']} | {s['mean_completion']:.3f} | "
            f"{s['collision_rate'] * 100:.0f}% | {s['red_light_violations']} ({s['red_light_violations_per_km']:.2f}/km) | "
            f"{(s['off_road_rate'] + s['deviation_rate']) * 100:.0f}% | {s['timeout_rate'] * 100:.0f}% |"
        )
    return "\n".join(lines)
