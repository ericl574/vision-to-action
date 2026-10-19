"""Fast unit checks for the camera-history and waypoint/control conventions."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from benchmarks.metadrive.model_controller import PurePursuitController, TemporalFrameHistory
from benchmarks.metadrive.prepare_trajectories import make_samples
from benchmarks.metadrive.run import is_safe_success


class PipelineTests(unittest.TestCase):
    def test_success_rejects_spawn_arrival_with_collision(self):
        self.assertFalse(is_safe_success(dict(arrive_dest=True, crash=True,
                                              out_of_road=True, route_completion=0.01)))
        self.assertTrue(is_safe_success(dict(arrive_dest=True, route_completion=0.98)))

    def test_history_keeps_half_second_spacing_at_every_step(self):
        history = TemporalFrameHistory(frames=4, sample_interval_steps=5)
        history.reset(np.zeros((2, 2, 3), dtype=np.uint8))
        for step in range(1, 17):
            history.update(np.full((2, 2, 3), step, dtype=np.uint8), step)
        self.assertEqual(history.as_array()[:, 0, 0, 0].tolist(), [1, 6, 11, 16])

    def test_right_waypoint_requires_negative_metadrive_steering(self):
        controller = PurePursuitController()
        forward = np.arange(1, 21, dtype=np.float64)
        right_path = np.stack((forward, np.ones(20)), axis=-1)
        left_path = np.stack((forward, -np.ones(20)), axis=-1)
        self.assertLess(controller.action(right_path, 5.0)[0], 0)
        self.assertGreater(controller.action(left_path, 5.0)[0], 0)

    def test_pose_targets_and_speed_units(self):
        with tempfile.TemporaryDirectory() as temp:
            episode = Path(temp) / "episode_1"
            episode.mkdir()
            with (episode / "actions.csv").open("w", newline="") as file:
                fields = ["image", "speed_mps", "position_x", "position_y", "heading_theta"]
                writer = csv.DictWriter(file, fieldnames=fields)
                writer.writeheader()
                for step in range(61):
                    writer.writerow(dict(image=f"frames/{step:06d}.jpg", speed_mps=5,
                                         position_x=step * 0.5, position_y=step * 0.1,
                                         heading_theta=0))
            rows = make_samples(episode, stride=5)
            self.assertEqual(len(rows), 3)
            self.assertEqual(float(rows[0]["speed"]), 5.0)
            import json
            first = np.asarray(json.loads(rows[0]["trajectory"]))
            np.testing.assert_allclose(first[-1], [25.0, -5.0], atol=1e-4)
            self.assertEqual(len(json.loads(rows[0]["history"])), 4)


if __name__ == "__main__":
    unittest.main()
