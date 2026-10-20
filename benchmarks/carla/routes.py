"""Route specs, route-following progress, and the high-level navigation command.

Nothing here imports ``carla``; the simulator-side code converts CARLA waypoints into plain
``(x, y)`` arrays and ``RoadOption`` integers before calling into this module, so the command
logic is unit-tested on any machine.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

FOLLOW, LEFT, RIGHT, STRAIGHT = 0, 1, 2, 3
COMMAND_NAMES = ("follow", "left", "right", "straight")
# agents.navigation.local_planner.RoadOption values -> our 4-way command.
CARLA_ROAD_OPTION_TO_COMMAND = {-1: FOLLOW, 1: LEFT, 2: RIGHT, 3: STRAIGHT, 4: FOLLOW, 5: FOLLOW, 6: FOLLOW}
LANE_CHANGE_OPTIONS = (5, 6)


@dataclass
class RouteSpec:
    town: str
    route_id: str
    start_idx: int
    end_idx: int
    weather: str = "ClearNoon"
    length_m: float = 0.0
    seed: int = 0


def save_routes(path, routes: list[RouteSpec]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump([asdict(route) for route in routes], handle, indent=1)


def load_routes(path) -> list[RouteSpec]:
    with open(path) as handle:
        return [RouteSpec(**item) for item in json.load(handle)]


class RouteTracker:
    """Track progress along a polyline route and expose the upcoming navigation command."""

    def __init__(self, xy, options, lookahead_m: float = 25.0, window: int = 40):
        self.xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        if len(self.xy) < 2:
            raise ValueError("A route needs at least two points")
        self.options = [int(option) for option in options]
        if len(self.options) != len(self.xy):
            raise ValueError("options must align with route points")
        segments = np.linalg.norm(np.diff(self.xy, axis=0), axis=1)
        self.cumulative = np.concatenate(([0.0], np.cumsum(segments)))
        self.length_m = float(self.cumulative[-1])
        self.lookahead_m = lookahead_m
        self.window = window
        self.index = 0
        self.completion = 0.0
        self.lateral_dev_m = 0.0

    def update(self, ego_xy) -> tuple[int, float, float]:
        ego = np.asarray(ego_xy, dtype=np.float64)
        lo = max(0, self.index - 5)  # a few points behind so lateral deviation is measured honestly
        hi = min(len(self.xy), self.index + self.window)
        distances = np.linalg.norm(self.xy[lo:hi] - ego, axis=1)
        nearest = lo + int(np.argmin(distances))
        self.index = max(self.index, nearest)  # progress never moves backwards
        self.lateral_dev_m = float(distances[nearest - lo])
        self.completion = float(self.cumulative[self.index] / self.length_m) if self.length_m > 0 else 1.0
        return self.index, self.completion, self.lateral_dev_m

    def command(self) -> int:
        horizon = self.cumulative[self.index] + self.lookahead_m
        for i in range(self.index, len(self.xy)):
            if self.cumulative[i] > horizon:
                break
            if self.options[i] != FOLLOW:
                return self.options[i]
        return FOLLOW

    @property
    def remaining_m(self) -> float:
        return float(self.length_m - self.cumulative[self.index])

    @property
    def end_xy(self) -> np.ndarray:
        return self.xy[-1]
