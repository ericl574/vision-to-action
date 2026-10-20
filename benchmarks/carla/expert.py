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

    def act(self, info: dict | None = None) -> dict:
        self._emergency = False
        control = self.agent.run_step(debug=False)
        target_kmh = float(getattr(self.agent._local_planner, "_target_speed", 0.0))
        target_mps = 0.0 if self._emergency else target_kmh / 3.6
        steer, throttle, brake = float(control.steer), float(control.throttle), float(control.brake)
        # BehaviorAgent notices crossing pedestrians late and releases the brake too early. The expert is
        # privileged, so it also stops for any in-lane vehicle/walker inside its stopping distance.
        if info is not None and info.get("hazard") and self._too_close(info):
            throttle, brake, target_mps = 0.0, 1.0, 0.0
            self._emergency = True
        return {"steer": steer, "throttle": throttle, "brake": brake,
                "target_speed_mps": target_mps, "emergency": int(self._emergency)}

    @staticmethod
    def _too_close(info: dict) -> bool:
        speed = float(info.get("speed_mps", 0.0))
        stopping = speed * speed / (2 * 4.0) + speed * 0.5  # 4 m/s^2 braking + 0.5 s reaction
        margin = 6.0 if info.get("hazard_type") == "walker" else 4.0
        return float(info.get("hazard_dist_m", 99.0)) < stopping + margin

    def perturbation(self, command: int = 0, steer: float = 0.0) -> tuple[float, bool]:
        """Triangular steering offset injected into the *executed* control (labels stay the expert's).

        New perturbations start only while lane-following on a near-straight road; starting one in a
        turn pushes the car onto the kerb and ends the route instead of teaching recovery.
        """
        if not self.perturb:
            return 0.0, False
        if self._perturb_left <= 0.0:
            if command == 0 and abs(steer) < 0.08 and self.rng.random() < self.rate * self.dt:
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
