"""Privileged teacher: CARLA's BehaviorAgent, instrumented to expose a desired-speed label.

The expert is used only to label states and to drive during collection. Lane changes and
tailgating manoeuvres are disabled so the student never has to imitate them.
"""

import random

from agents.navigation.behavior_agent import BehaviorAgent


class Expert:
    def __init__(self, env, behavior: str = "normal", perturb: bool = True, seed: int = 0,
                 perturb_rate_per_s: float = 0.1, perturb_duration_s: float = 1.5, perturb_amplitude: float = 0.25,
                 dt: float = 0.1):
        try:
            self.agent = BehaviorAgent(env.vehicle, behavior=behavior, map_inst=env.map, grp_inst=env.grp)
        except TypeError:  # older signature
            self.agent = BehaviorAgent(env.vehicle, behavior=behavior)
        self.agent.set_global_plan(env.plan, stop_waypoint_creation=True, clean_queue=True)
        self.agent._tailgating = lambda *args, **kwargs: None  # no overtaking / lane changes
        original_stop = self.agent.emergency_stop

        def instrumented_stop():
            self._emergency = True
            return original_stop()

        self.agent.emergency_stop = instrumented_stop
        self._emergency = False
        self.rng = random.Random(seed)
        self.perturb = perturb
        self.rate, self.duration, self.amplitude, self.dt = perturb_rate_per_s, perturb_duration_s, perturb_amplitude, dt
        self._perturb_left = 0.0
        self._perturb_sign = 1.0
        self._perturb_total = 0.0

    def act(self) -> dict:
        self._emergency = False
        control = self.agent.run_step(debug=False)
        target_kmh = float(getattr(self.agent._local_planner, "_target_speed", 0.0))
        target_mps = 0.0 if self._emergency else target_kmh / 3.6
        return {"steer": float(control.steer), "throttle": float(control.throttle), "brake": float(control.brake),
                "target_speed_mps": target_mps, "emergency": int(self._emergency)}

    def perturbation(self) -> tuple[float, bool]:
        """Triangular steering offset injected into the *executed* control (labels stay the expert's)."""
        if not self.perturb:
            return 0.0, False
        if self._perturb_left <= 0.0:
            if self.rng.random() < self.rate * self.dt:
                self._perturb_left = self._perturb_total = self.duration
                self._perturb_sign = self.rng.choice((-1.0, 1.0))
            else:
                return 0.0, False
        progress = 1.0 - self._perturb_left / self._perturb_total
        magnitude = self.amplitude * (1.0 - abs(2.0 * progress - 1.0))
        self._perturb_left -= self.dt
        return self._perturb_sign * magnitude, True

    def done(self) -> bool:
        return self.agent.done()
