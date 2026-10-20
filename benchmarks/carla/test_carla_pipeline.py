"""Mac-runnable tests for everything that does not need the CARLA simulator.

    conda run -n metadrive-ai python -m unittest benchmarks.carla.test_carla_pipeline -v
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.carla import metrics  # noqa: E402
from benchmarks.carla.dataset import build_samples, history_indices, sample_weights  # noqa: E402
from benchmarks.carla.model import FrameHistory, SpeedServo, create_network  # noqa: E402
from benchmarks.carla.record import RouteRecorder, load_manifest  # noqa: E402
from benchmarks.carla.routes import FOLLOW, LEFT, RouteSpec, RouteTracker, load_routes, save_routes  # noqa: E402


def straight_route_with_left_turn(length_m=100.0, spacing=2.0, turn_from=40.0, turn_to=50.0):
    xs = np.arange(0.0, length_m + spacing, spacing)
    xy = np.stack([xs, np.zeros_like(xs)], axis=1)
    options = [LEFT if turn_from <= x <= turn_to else FOLLOW for x in xs]
    return xy, options


class RouteTests(unittest.TestCase):
    def test_command_switches_before_junction_and_back_after(self):
        xy, options = straight_route_with_left_turn()
        tracker = RouteTracker(xy, options, lookahead_m=25.0)
        tracker.update((10.0, 0.0))
        self.assertEqual(tracker.command(), FOLLOW, "40 m ahead is beyond the 25 m look-ahead")
        tracker.update((20.0, 0.0))
        self.assertEqual(tracker.command(), LEFT, "turn starts within 25 m -> announce it")
        tracker.update((45.0, 0.0))
        self.assertEqual(tracker.command(), LEFT)
        tracker.update((60.0, 0.0))
        self.assertEqual(tracker.command(), FOLLOW)

    def test_progress_is_monotonic_and_reports_lateral_deviation(self):
        xy, options = straight_route_with_left_turn()
        tracker = RouteTracker(xy, options)
        _, c1, _ = tracker.update((30.0, 0.0))
        _, c2, dev = tracker.update((28.0, 4.0))  # moved backwards and sideways
        self.assertGreaterEqual(c2, c1)
        self.assertAlmostEqual(dev, 4.0, places=5)
        _, c3, _ = tracker.update((100.0, 0.0))
        self.assertAlmostEqual(c3, 1.0, places=5)
        self.assertAlmostEqual(tracker.remaining_m, 0.0, places=5)

    def test_routes_roundtrip_json(self):
        routes = [RouteSpec("Town10HD_Opt", "t10_000", 3, 17, "WetNoon", 412.5, 7)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "routes" / "dev.json"
            save_routes(path, routes)
            self.assertEqual(load_routes(path), routes)


class RecordTests(unittest.TestCase):
    def _frame(self, value):
        return np.full((8, 16, 3), value, dtype=np.uint8)

    def test_finish_writes_labels_and_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = RouteRecorder(tmp, "r1", {"town": "Town10HD_Opt"})
            for i in range(3):
                recorder.append(self._frame(i), {"t": i / 10, "speed_mps": 1.0, "command": 0, "steer": 0.1,
                                                 "target_speed_mps": 7.0, "tl_state": 0, "hazard": 0})
            recorder.finish("arrived", {"distance_m": 12.0})
            manifest = load_manifest(tmp)
            self.assertEqual(len(manifest), 1)
            self.assertEqual(manifest[0]["status"], "arrived")
            self.assertEqual(manifest[0]["n_frames"], 3)
            self.assertTrue((Path(tmp) / "r1" / "frames" / "000002.jpg").exists())
            with open(Path(tmp) / "r1" / "labels.csv") as handle:
                self.assertEqual(len(handle.read().strip().splitlines()), 4)

    def test_manifest_skips_interrupted_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            done = RouteRecorder(tmp, "done", {})
            done.append(self._frame(1), {"t": 0.0})
            done.finish("arrived")
            partial = RouteRecorder(tmp, "partial", {})
            partial.append(self._frame(1), {"t": 0.0})  # never finished
            self.assertEqual([m["route_id"] for m in load_manifest(tmp)], ["done"])


class MetricsTests(unittest.TestCase):
    def test_red_light_violation_disqualifies_success(self):
        clean = metrics.route_outcome("arrived", 0.99, red_light_violations=0)
        dirty = metrics.route_outcome("arrived", 0.99, red_light_violations=1)
        self.assertTrue(clean["success"])
        self.assertFalse(dirty["success"])

    def test_success_requires_95_percent_completion(self):
        self.assertFalse(metrics.route_outcome("arrived", 0.90, 0)["success"])
        self.assertTrue(metrics.route_outcome("arrived", 0.95, 0)["success"])
        self.assertFalse(metrics.route_outcome("timeout", 0.99, 0)["success"])

    def test_monitor_counts_crossing_on_red_but_not_other_lane_or_stopped_then_green(self):
        monitor = metrics.RedLightMonitor()
        # Red light for another lane: ego is never "at" it, so nothing is counted even if it is red.
        self.assertFalse(monitor.update(False, 7, True, 8.0))
        # Enter own-lane trigger box on red, leave while still moving -> violation.
        monitor.update(True, 1, True, 8.0)
        self.assertTrue(monitor.update(False, 1, True, 7.5))
        # Enter on red, stop, light turns green, then leave -> no violation.
        monitor.update(True, 2, True, 3.0)
        monitor.update(True, 2, True, 0.0)
        monitor.update(True, 2, False, 0.0)
        self.assertFalse(monitor.update(False, 2, False, 2.0))
        # Enter on green, light turns red while crossing, leave moving -> committed crossing, not a violation.
        monitor.update(True, 3, False, 7.0)
        monitor.update(True, 3, True, 7.0)
        self.assertFalse(monitor.update(False, 3, True, 7.0))
        self.assertEqual(monitor.violations, 1)

    def test_termination_prefers_collision_over_arrival(self):
        self.assertEqual(metrics.check_termination(10, 100, "vehicle", 0, 0, True, 0), "collision")
        self.assertEqual(metrics.check_termination(10, 100, None, 0, 0, True, 0), "arrived")
        self.assertEqual(metrics.check_termination(10, 100, None, 10, 0, False, 0), "off_road")
        self.assertEqual(metrics.check_termination(10, 100, None, 0, 10, False, 0), "route_deviation")
        self.assertIsNone(metrics.check_termination(10, 100, None, 3, 3, False, 0))
        self.assertEqual(metrics.check_termination(99, 100, None, 0, 0, False, 0), "timeout")

    def test_summary_and_table(self):
        rows = [
            {"status": "arrived", "route_completion": 1.0, "collision_type": "", "red_light_violations": 0,
             "success": True, "distance_m": 500},
            {"status": "collision", "route_completion": 0.4, "collision_type": "walker", "red_light_violations": 1,
             "success": False, "distance_m": 200},
        ]
        summary = metrics.summarize(rows)
        self.assertEqual(summary["successes"], 1)
        self.assertAlmostEqual(summary["collision_walker_rate"], 0.5)
        self.assertAlmostEqual(summary["red_light_violations_per_km"], 1 / 0.7)
        table = metrics.markdown_table({"pilot v1": summary})
        self.assertIn("| pilot v1 | 1/2 |", table)


def make_fake_dataset(root, n_routes=4, frames=12, status="arrived"):
    for r in range(n_routes):
        recorder = RouteRecorder(root, f"route_{r}", {"town": "Town10HD_Opt"})
        for i in range(frames):
            braking = i % 4 == 0
            recorder.append(np.full((9, 16, 3), i * 10, dtype=np.uint8), {
                "t": i / 10, "speed_mps": 5.0, "command": r % 4, "steer": 0.3 if r % 2 else 0.0,
                "target_speed_mps": 0.0 if braking else 7.0, "tl_state": 3 if braking else 1,
                "hazard": 0, "hazard_dist_m": 0.0})
        recorder.finish(status)


class DatasetTests(unittest.TestCase):
    def test_history_pads_with_first_frame(self):
        self.assertEqual(history_indices(0, 3, 3), [0, 0, 0])
        self.assertEqual(history_indices(4, 3, 3), [0, 1, 4])
        self.assertEqual(history_indices(10, 3, 3), [4, 7, 10])

    def test_split_is_by_route_and_sampler_upweights_braking(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_fake_dataset(tmp, n_routes=4, frames=12)
            train, val, split = build_samples([tmp], history=3, stride=3, val_frac=0.25, seed=1)
            self.assertEqual(len(split["val"]), 1)
            self.assertTrue(set(r["route_id"] for r in train).isdisjoint(set(r["route_id"] for r in val)))
            self.assertEqual(len(train) + len(val), 48)
            weights = sample_weights(train)
            braking = [w for w, r in zip(weights, train) if r["target_speed"] < 1.0]
            cruising = [w for w, r in zip(weights, train) if r["target_speed"] >= 1.0]
            self.assertGreater(min(braking), max(cruising))

    def test_saved_split_is_reused_and_new_routes_go_to_train(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_fake_dataset(Path(tmp) / "a", n_routes=4)
            _, _, split = build_samples([str(Path(tmp) / "a")], val_frac=0.25, seed=1)
            split_path = Path(tmp) / "split.json"
            split_path.write_text(json.dumps(split))
            make_fake_dataset(Path(tmp) / "b", n_routes=2)
            # route ids collide across roots in this fake; rename b's routes to be distinct
            for r in range(2):
                meta = Path(tmp) / "b" / f"route_{r}" / "meta.json"
                data = json.loads(meta.read_text())
                data["route_id"] = f"dagger_{r}"
                meta.write_text(json.dumps(data))
            train, val, split2 = build_samples([str(Path(tmp) / "a"), str(Path(tmp) / "b")], split_file=split_path)
            self.assertEqual(split2["val"], split["val"])
            self.assertTrue(all(r["route_id"].startswith("dagger") or r["route_id"] in split["train"] for r in train))

    def test_crashed_route_drops_its_last_second(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_fake_dataset(tmp, n_routes=1, frames=25, status="collision")
            train, val, _ = build_samples([tmp], val_frac=0.0)
            self.assertEqual(len(train) + len(val), 15)


class ModelTests(unittest.TestCase):
    def test_forward_shapes_on_cpu(self):
        import torch

        model = create_network(history=3, pretrained=False, backbone="resnet18")
        model.eval()
        images = torch.rand(2, 3, 3, 144, 256)
        out = model(images, torch.tensor([[5.0], [0.0]]), torch.tensor([0, 2]))
        self.assertEqual(tuple(out["steer"].shape), (2,))
        self.assertEqual(tuple(out["target_speed"].shape), (2,))
        self.assertEqual(tuple(out["tl_logits"].shape), (2, 4))
        self.assertTrue(bool((out["target_speed"] >= 0).all()))
        self.assertTrue(bool((out["steer"].abs() <= 1).all()))

    def test_speed_servo(self):
        servo = SpeedServo()
        throttle, brake = servo(7.0, 3.0)
        self.assertGreater(throttle, 0)
        self.assertEqual(brake, 0)
        throttle, brake = servo(0.0, 7.0)
        self.assertEqual(throttle, 0)
        self.assertEqual(brake, 1.0)
        self.assertEqual(servo(0.0, 0.0), (0.0, 1.0), "hold the brake at a standstill")

    def test_frame_history_order_and_padding(self):
        history = FrameHistory(history=3, stride=2)
        frames = [np.full((2, 2, 3), i, dtype=np.uint8) for i in range(6)]
        history.reset(frames[0])
        self.assertEqual([int(f[0, 0, 0]) for f in history.stack()], [0, 0, 0])
        for frame in frames[1:]:
            history.push(frame)
        self.assertEqual([int(f[0, 0, 0]) for f in history.stack()], [1, 3, 5])


class FakeEnv:
    """Scripted stand-in for CarlaEnv: drives straight at 5 m/s, optional events by step."""

    fixed_dt = 0.1

    def __init__(self, events=None, route_length=200.0):
        self.events = events or {}
        self.route_length = route_length
        self.tracker = type("T", (), {"length_m": route_length})()
        self.step_i = 0

    def _info(self):
        x = self.step_i * 0.5
        info = {"x": x, "y": 0.0, "yaw": 0.0, "speed_mps": 5.0, "tl_state": 0, "hazard": 0, "collision_type": None,
                "collision_other": "", "off_road": False, "deviated": False, "arrived": x >= self.route_length - 6,
                "route_completion": min(1.0, x / self.route_length), "red_light_violations": 0, "t": self.step_i * 0.1}
        info.update(self.events.get(self.step_i, {}))
        return info

    def reset(self, spec):
        self.step_i = 0
        return {"rgb": np.zeros((4, 8, 3), np.uint8), "speed_mps": 0.0, "command": 0}, self._info()

    def step(self, steer, throttle, brake):
        self.step_i += 1
        return {"rgb": np.zeros((4, 8, 3), np.uint8), "speed_mps": 5.0, "command": 0}, self._info()


class EpisodeTests(unittest.TestCase):
    def test_arrival_is_a_success_and_distance_is_integrated(self):
        from benchmarks.carla.episode import run_episode

        spec = RouteSpec("Town10HD_Opt", "r", 0, 1)
        result = run_episode(FakeEnv(), spec, lambda obs, info, step: (0.0, 0.5, 0.0, None), max_steps=1000)
        self.assertEqual(result["status"], "arrived")
        self.assertTrue(result["success"])
        self.assertAlmostEqual(result["distance_m"], result["steps"] * 0.5, places=1)

    def test_collision_ends_the_route_and_red_light_breaks_success(self):
        from benchmarks.carla.episode import run_episode

        spec = RouteSpec("Town10HD_Opt", "r", 0, 1)
        env = FakeEnv(events={20: {"collision_type": "walker", "collision_other": "walker.pedestrian.0001"}})
        result = run_episode(env, spec, lambda obs, info, step: (0.0, 0.5, 0.0, None), max_steps=1000)
        self.assertEqual((result["status"], result["collision_type"], result["steps"]), ("collision", "walker", 20))
        env = FakeEnv(events={i: {"red_light_violations": 1} for i in range(30, 1000)})
        result = run_episode(env, spec, lambda obs, info, step: (0.0, 0.5, 0.0, None), max_steps=1000)
        self.assertEqual(result["status"], "arrived")
        self.assertFalse(result["success"])

    def test_on_step_sees_pre_control_state(self):
        from benchmarks.carla.episode import run_episode

        seen = []
        run_episode(FakeEnv(route_length=20), RouteSpec("T", "r", 0, 1),
                    lambda obs, info, step: (0.0, 0.0, 0.0, {"k": step}),
                    max_steps=10, on_step=lambda obs, info, step, control, extra: seen.append((step, info["x"], extra["k"])))
        self.assertEqual(seen[0], (0, 0.0, 0))
        self.assertEqual(seen[1], (1, 0.5, 1))


class HudTests(unittest.TestCase):
    def test_render_video_from_logged_outputs(self):
        from benchmarks.carla import hud
        from benchmarks.carla.record import OUTPUT_COLUMNS

        with tempfile.TemporaryDirectory() as tmp:
            recorder = RouteRecorder(tmp, "clip", {}, columns=OUTPUT_COLUMNS, csv_name="outputs.csv")
            for i in range(5):
                recorder.append(np.random.randint(0, 255, (72, 128, 3), dtype=np.uint8), {
                    "t": i / 10, "speed_mps": 4.0, "command": 2, "steer": 0.1, "throttle": 0.3, "brake": 0.0,
                    "target_speed_mps": 6.0, "tl_none": 0.9, "tl_green": 0.05, "tl_yellow": 0.03, "tl_red": 0.02,
                    "hazard_prob": 0.1, "hazard_dist_m": 0.0})
            recorder.finish("arrived")
            out = hud.render_video(Path(tmp) / "clip", Path(tmp) / "clip.mp4", fps=10)
            self.assertTrue(out.exists() and out.stat().st_size > 0)
            rows, ground_truth = hud.load_log(Path(tmp) / "clip")
            self.assertFalse(ground_truth)
            self.assertEqual(len(rows), 5)

    def test_render_frame_shape_and_caption(self):
        from benchmarks.carla import hud

        rgb = np.random.randint(0, 255, (288, 512, 3), dtype=np.uint8)
        out = {"steer": -0.3, "speed_mps": 6.0, "target_speed_mps": 0.0, "command": 1,
               "tl_probs": [0.02, 0.03, 0.05, 0.9], "hazard_prob": 0.1, "hazard_dist_m": 0.0, "t": 4.2}
        frame = hud.render_frame(rgb, out)
        self.assertEqual(frame.shape, (hud.CAM_H + hud.PANEL_H, hud.CANVAS_W, 3))
        self.assertEqual(hud.decision_caption(0.0, 6.0, [0.02, 0.03, 0.05, 0.9], 0.1), "STOPPING: network sees red light")
        self.assertTrue(hud.decision_caption(7.0, 6.5, [1, 0, 0, 0], 0.0).startswith("CRUISING"))


if __name__ == "__main__":
    unittest.main()
