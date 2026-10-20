"""Thin CARLA wrapper: synchronous world, ego + front camera, seeded traffic, route tracking.

``obs`` is what a student may see: ``{"rgb", "speed_mps", "command"}``.
``info`` is privileged (labels and metrics only): traffic-light state for the ego lane, hazards,
collisions, off-road, red-light violations, route progress.
"""

import math
import queue
import random

import numpy as np

import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner

from benchmarks.carla.metrics import RedLightMonitor
from benchmarks.carla.routes import CARLA_ROAD_OPTION_TO_COMMAND, LANE_CHANGE_OPTIONS, FOLLOW, RouteSpec, RouteTracker

TL_STATE = {carla.TrafficLightState.Green: 1, carla.TrafficLightState.Yellow: 2, carla.TrafficLightState.Red: 3}
DEVIATION_M = 6.0


def weather_by_name(name: str):
    return getattr(carla.WeatherParameters, name)


class CarlaEnv:
    def __init__(self, client, image_w: int = 512, image_h: int = 288, fov: float = 100.0, fixed_dt: float = 0.1,
                 vehicles: int = 40, walkers: int = 40, tm_port: int = 8000, hazard_range_m: float = 25.0,
                 tl_range_m: float = 40.0, traffic_radius_m: float = 160.0):
        self.client = client
        self.image_w, self.image_h, self.fov, self.fixed_dt = image_w, image_h, fov, fixed_dt
        self.n_vehicles, self.n_walkers = vehicles, walkers
        self.hazard_range_m, self.tl_range_m, self.traffic_radius_m = hazard_range_m, tl_range_m, traffic_radius_m
        self.tm = client.get_trafficmanager(tm_port)
        self.tm_port = tm_port
        self.world = None
        self.map = None
        self.grp = None
        self.town = None
        self.spawn_points = []
        self.vehicle = None
        self.plan = []
        self.tracker = None
        self._traffic = []
        self._walker_controllers = []
        self._sensors = []
        self._images: queue.Queue = queue.Queue()
        self._collision = None
        self._red = RedLightMonitor()
        self._step = 0
        self._last_obs = None

    # ------------------------------------------------------------------ world
    def load_town(self, town: str) -> None:
        if self.world is None or self.town != town:
            self.world = self.client.load_world(town)
            self.town = town
            self.map = self.world.get_map()
            self.grp = GlobalRoutePlanner(self.map, 2.0)
            self.spawn_points = self.map.get_spawn_points()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.fixed_dt
        self.world.apply_settings(settings)
        self.tm.set_synchronous_mode(True)
        self.tm.set_hybrid_physics_mode(True)
        self.tm.set_hybrid_physics_radius(70.0)
        self.tm.set_global_distance_to_leading_vehicle(3.0)

    def trace(self, start_idx: int, end_idx: int):
        start = self.spawn_points[start_idx].location
        end = self.spawn_points[end_idx].location
        return self.grp.trace_route(start, end)

    @staticmethod
    def plan_geometry(plan):
        xy = np.array([[wp.transform.location.x, wp.transform.location.y] for wp, _ in plan])
        options = [CARLA_ROAD_OPTION_TO_COMMAND.get(int(option.value), FOLLOW) for _, option in plan]
        return xy, options

    @staticmethod
    def plan_length_m(plan) -> float:
        xy, _ = CarlaEnv.plan_geometry(plan)
        return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()) if len(xy) > 1 else 0.0

    @staticmethod
    def plan_has_lane_change(plan) -> bool:
        return any(int(option.value) in LANE_CHANGE_OPTIONS for _, option in plan)

    @staticmethod
    def plan_turns(plan) -> int:
        """Number of junction manoeuvres (runs of LEFT/RIGHT/STRAIGHT)."""
        turns, previous = 0, FOLLOW
        for _, option in plan:
            command = CARLA_ROAD_OPTION_TO_COMMAND.get(int(option.value), FOLLOW)
            if command != FOLLOW and previous == FOLLOW:
                turns += 1
            previous = command
        return turns

    # ------------------------------------------------------------------ episode
    def reset(self, spec: RouteSpec):
        self.cleanup()
        self.load_town(spec.town)
        self.world.set_weather(weather_by_name(spec.weather))
        rng = random.Random(spec.seed)
        self.tm.set_random_device_seed(spec.seed)
        self.plan = self.trace(spec.start_idx, spec.end_idx)
        xy, options = self.plan_geometry(self.plan)
        self.tracker = RouteTracker(xy, options)

        bp_lib = self.world.get_blueprint_library()
        ego_bp = bp_lib.find("vehicle.tesla.model3")
        ego_bp.set_attribute("role_name", "hero")
        spawn = self.spawn_points[spec.start_idx]
        self.vehicle = self.world.try_spawn_actor(ego_bp, spawn)
        if self.vehicle is None:
            raise RuntimeError(f"Could not spawn ego at spawn point {spec.start_idx} in {spec.town}")

        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(self.image_w))
        cam_bp.set_attribute("image_size_y", str(self.image_h))
        cam_bp.set_attribute("fov", str(self.fov))
        camera = self.world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.3, z=1.7)), attach_to=self.vehicle)
        self._images = queue.Queue()
        camera.listen(self._images.put)
        collision = self.world.spawn_actor(bp_lib.find("sensor.other.collision"), carla.Transform(), attach_to=self.vehicle)
        collision.listen(self._on_collision)
        self._sensors = [camera, collision]
        self._collision = None
        self._red = RedLightMonitor()
        self._step = 0

        self._spawn_traffic(rng, xy, spawn.location)
        for _ in range(5):  # settle physics and fill the camera queue
            self.world.tick()
            self._drain_images()
        obs = self._observe(tick_already_done=False)
        return obs, self._privileged()

    def step(self, steer: float, throttle: float, brake: float):
        self.vehicle.apply_control(carla.VehicleControl(
            throttle=float(np.clip(throttle, 0.0, 1.0)), steer=float(np.clip(steer, -1.0, 1.0)),
            brake=float(np.clip(brake, 0.0, 1.0))))
        self.world.tick()
        self._step += 1
        obs = self._observe(tick_already_done=True)
        return obs, self._privileged()

    # ------------------------------------------------------------------ internals
    def _on_collision(self, event):
        if self._collision is not None:
            return
        type_id = event.other_actor.type_id if event.other_actor is not None else "static"
        if type_id.startswith("vehicle."):
            kind = "vehicle"
        elif type_id.startswith("walker."):
            kind = "walker"
        else:
            kind = "static"
        impulse = event.normal_impulse
        self._collision = {"type": kind, "other": type_id,
                           "impulse": float(math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2))}

    def _drain_images(self):
        latest = None
        while True:
            try:
                latest = self._images.get_nowait()
            except queue.Empty:
                return latest

    def _observe(self, tick_already_done: bool):
        frame_id = self.world.get_snapshot().frame
        image = None
        deadline = 50
        while deadline > 0:
            image = self._images.get(timeout=10.0)
            if image.frame >= frame_id:
                break
            deadline -= 1
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
        rgb = np.ascontiguousarray(array[:, :, :3][:, :, ::-1])
        speed = self.speed_mps()
        location = self.vehicle.get_location()
        self.tracker.update((location.x, location.y))
        obs = {"rgb": rgb, "speed_mps": speed, "command": self.tracker.command(), "frame": int(image.frame)}
        self._last_obs = obs
        return obs

    def speed_mps(self) -> float:
        v = self.vehicle.get_velocity()
        return float(math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))

    def _privileged(self) -> dict:
        transform = self.vehicle.get_transform()
        location = transform.location
        speed = self.speed_mps()
        ego_wp = self.map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)

        at_light = self.vehicle.is_at_traffic_light()
        light = self.vehicle.get_traffic_light() if at_light else None
        is_red = bool(light is not None and light.get_state() == carla.TrafficLightState.Red)
        ran_red = self._red.update(at_light, light.id if light is not None else None, is_red, speed)

        tl_state, tl_dist = self._traffic_light_ahead(ego_wp, location, light)
        hazard, hazard_dist, hazard_type = self._hazard_ahead(transform)

        drivable = self.map.get_waypoint(location, project_to_road=False,
                                         lane_type=carla.LaneType.Driving | carla.LaneType.Parking | carla.LaneType.Bidirectional)
        off_road = drivable is None
        end = self.tracker.end_xy
        dist_to_end = float(math.hypot(location.x - end[0], location.y - end[1]))
        # Routes can pass close to their own endpoint early on, so arrival also needs route progress.
        arrived = self.tracker.remaining_m < 12.0 and dist_to_end < 15.0
        return {
            "speed_mps": speed,
            "tl_state": tl_state,
            "tl_dist_m": tl_dist,
            "hazard": int(hazard),
            "hazard_dist_m": hazard_dist,
            "hazard_type": hazard_type,
            "collision_type": self._collision["type"] if self._collision else None,
            "collision_other": self._collision["other"] if self._collision else "",
            "off_road": bool(off_road),
            "ran_red_light": bool(ran_red),
            "red_light_violations": self._red.violations,
            "route_completion": self.tracker.completion,
            "lateral_dev_m": self.tracker.lateral_dev_m,
            "deviated": self.tracker.lateral_dev_m > DEVIATION_M,
            "arrived": bool(arrived),
            "x": location.x, "y": location.y, "yaw": transform.rotation.yaw,
            "step": self._step,
            "t": self._step * self.fixed_dt,
        }

    def _traffic_light_ahead(self, ego_wp, location, current_light):
        """State of the light governing the ego lane within ``tl_range_m`` (0 none, 1 green, 2 yellow, 3 red)."""
        light, distance = None, 0.0
        try:
            candidates = self.world.get_traffic_lights_from_waypoint(ego_wp, self.tl_range_m)
        except Exception:  # noqa: BLE001 - API not available
            candidates = []
        if candidates:
            light = candidates[0]
            stops = light.get_stop_waypoints()
            if stops:
                distance = min(location.distance(wp.transform.location) for wp in stops)
            else:
                distance = location.distance(light.get_location())
        elif current_light is not None:
            light, distance = current_light, 0.0
        if light is None:
            return 0, 0.0
        return TL_STATE.get(light.get_state(), 0), float(distance)

    def _hazard_ahead(self, transform):
        """Nearest vehicle/walker inside a forward corridor (privileged label only)."""
        yaw = math.radians(transform.rotation.yaw)
        fx, fy = math.cos(yaw), math.sin(yaw)
        ex, ey = transform.location.x, transform.location.y
        best = (False, 0.0, "")
        best_dist = self.hazard_range_m
        for actor in self.world.get_actors().filter("*vehicle*"):
            if actor.id == self.vehicle.id:
                continue
            loc = actor.get_location()
            dx, dy = loc.x - ex, loc.y - ey
            forward = dx * fx + dy * fy
            lateral = abs(-dx * fy + dy * fx)
            if 0.0 < forward < best_dist and lateral < 2.2:
                best_dist = forward
                best = (True, max(0.0, forward - 4.5), "vehicle")
        for actor in self.world.get_actors().filter("walker.pedestrian.*"):
            loc = actor.get_location()
            dx, dy = loc.x - ex, loc.y - ey
            forward = dx * fx + dy * fy
            lateral = abs(-dx * fy + dy * fx)
            if 0.0 < forward < min(best_dist, 20.0) and lateral < 2.4:
                best_dist = forward
                best = (True, max(0.0, forward - 2.5), "walker")
        return best

    def _spawn_traffic(self, rng: random.Random, route_xy: np.ndarray, ego_location) -> None:
        bp_lib = self.world.get_blueprint_library()
        vehicle_bps = [bp for bp in bp_lib.filter("vehicle.*")
                       if int(bp.get_attribute("number_of_wheels")) == 4 and not bp.id.startswith("vehicle.carlamotors")]
        walker_bps = list(bp_lib.filter("walker.pedestrian.*"))
        controller_bp = bp_lib.find("controller.ai.walker")

        def near_route(location, radius):
            d = np.linalg.norm(route_xy - np.array([location.x, location.y]), axis=1)
            return float(d.min()) < radius

        # Vehicles on spawn points close to the route, away from the ego.
        candidates = [sp for sp in self.spawn_points
                      if sp.location.distance(ego_location) > 12.0 and near_route(sp.location, self.traffic_radius_m)]
        rng.shuffle(candidates)
        batch = []
        for sp in candidates[: self.n_vehicles]:
            bp = rng.choice(vehicle_bps)
            if bp.has_attribute("color"):
                bp.set_attribute("color", rng.choice(bp.get_attribute("color").recommended_values))
            bp.set_attribute("role_name", "autopilot")
            batch.append(carla.command.SpawnActor(bp, sp).then(carla.command.SetAutopilot(carla.command.FutureActor, True, self.tm_port)))
        self._traffic = []
        for response in self.client.apply_batch_sync(batch, True):
            if not response.error:
                self._traffic.append(response.actor_id)

        # Walkers near the route; half of them may cross roads.
        self.world.set_pedestrians_cross_factor(0.5)
        self.world.set_pedestrians_seed(rng.randrange(1 << 30))
        spawn_batch, speeds = [], []
        attempts = 0
        while len(spawn_batch) < self.n_walkers and attempts < self.n_walkers * 25:
            attempts += 1
            loc = self.world.get_random_location_from_navigation()
            if loc is None or loc.distance(ego_location) < 10.0 or not near_route(loc, 60.0):
                continue
            bp = rng.choice(walker_bps)
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")
            spawn_batch.append(carla.command.SpawnActor(bp, carla.Transform(loc)))
            speeds.append(1.0 + rng.random() * 0.8)
        walker_ids = [r.actor_id for r in self.client.apply_batch_sync(spawn_batch, True) if not r.error]
        controller_batch = [carla.command.SpawnActor(controller_bp, carla.Transform(), wid) for wid in walker_ids]
        controller_ids = [r.actor_id for r in self.client.apply_batch_sync(controller_batch, True) if not r.error]
        self.world.tick()
        self._walker_controllers = []
        for cid in controller_ids:
            controller = self.world.get_actor(cid)
            if controller is None:
                continue
            controller.start()
            target = self.world.get_random_location_from_navigation()
            if target is not None:
                controller.go_to_location(target)
            controller.set_max_speed(speeds[len(self._walker_controllers) % len(speeds)] if speeds else 1.2)
            self._walker_controllers.append(cid)
        self._traffic.extend(walker_ids)

    def cleanup(self) -> None:
        if self.world is None:
            return
        for sensor in self._sensors:
            try:
                sensor.stop()
                sensor.destroy()
            except Exception:  # noqa: BLE001
                pass
        self._sensors = []
        for cid in self._walker_controllers:
            controller = self.world.get_actor(cid)
            if controller is not None:
                try:
                    controller.stop()
                except Exception:  # noqa: BLE001
                    pass
        ids = list(self._walker_controllers) + list(self._traffic)
        if self.vehicle is not None:
            ids.append(self.vehicle.id)
        if ids:
            self.client.apply_batch_sync([carla.command.DestroyActor(i) for i in ids], True)
        self._walker_controllers, self._traffic, self.vehicle = [], [], None
        try:
            self.world.tick()
        except Exception:  # noqa: BLE001
            pass

    def close(self) -> None:
        self.cleanup()
        if self.world is not None:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
            self.tm.set_synchronous_mode(False)
