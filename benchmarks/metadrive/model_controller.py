"""Camera-history model inference and simple trajectory-following control.

The learned model predicts points in the same ego frame used by
``prepare_comma1m.py``: x is forward and y is right. This module contains no
MetaDrive imports so the model and controller can be unit-tested separately
from Panda3D.
"""

from collections import deque
from pathlib import Path

import numpy as np


class ResidualTrajectoryModel:
    """Load ``ResidualCNNGRU`` and predict a 20-point ego-relative trajectory."""

    def __init__(self, checkpoint: Path, device: str = "auto"):
        import torch

        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Trajectory checkpoint not found: {checkpoint}")

        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)

        # Import lazily so zero-steering benchmark runs do not require torch.
        from train_comma1m_v2 import ResidualCNNGRU

        self.model = ResidualCNNGRU().to(self.device)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("model_state_dict", payload)
        self.model.load_state_dict(state)
        self.model.eval()
        self.epoch = payload.get("epoch")
        self.validation_ade = payload.get("val_ade")
        self.checkpoint = checkpoint

    def predict(self, frames_rgb: np.ndarray, speed_mps: float) -> np.ndarray:
        import torch

        frames = np.asarray(frames_rgb)
        if frames.ndim != 4 or frames.shape[0] != 4 or frames.shape[3] != 3:
            raise ValueError(f"Expected four RGB frames, got shape {frames.shape}")
        if frames.dtype != np.uint8:
            raise ValueError(f"Expected uint8 camera frames, got {frames.dtype}")
        if not np.isfinite(speed_mps):
            raise ValueError(f"Non-finite speed: {speed_mps}")

        tensor = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).float().div_(255.0)
        tensor = tensor.unsqueeze(0).to(self.device)
        speed = torch.tensor([[float(speed_mps) / 30.0]], dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            prediction, _baseline = self.model(tensor, speed)
        trajectory = prediction[0].detach().float().cpu().numpy()
        if trajectory.shape != (20, 2) or not np.isfinite(trajectory).all():
            raise ValueError(f"Model emitted an invalid trajectory: {trajectory.shape}")
        return trajectory


class TemporalFrameHistory:
    """Keep four frames spaced like the training samples.

    MetaDrive's default step is 5 x 0.02 s = 0.1 s. comma1M histories use
    adjacent saved frames 0.5 s apart. Keep every simulator frame so inference
    can update at 10 Hz while its four inputs remain 0.5 s apart.
    """

    def __init__(self, frames: int = 4, sample_interval_steps: int = 5):
        if frames < 1 or sample_interval_steps < 1:
            raise ValueError("frames and sample_interval_steps must be positive")
        self.frames = frames
        self.sample_interval_steps = sample_interval_steps
        self._history = deque(maxlen=(frames - 1) * sample_interval_steps + 1)

    def reset(self, frame_rgb: np.ndarray) -> None:
        frame = self._validate(frame_rgb)
        self._history.clear()
        for _ in range(self._history.maxlen):
            self._history.append(frame.copy())

    def update(self, frame_rgb: np.ndarray, step: int) -> bool:
        """Add a frame when due and return whether inference should run."""
        frame = self._validate(frame_rgb)
        if not self._history:
            self.reset(frame)
            return True
        self._history.append(frame.copy())
        return True

    @property
    def ready(self) -> bool:
        return len(self._history) == self._history.maxlen

    def as_array(self) -> np.ndarray:
        if not self.ready:
            raise RuntimeError("Frame history is not initialized")
        past = tuple(self._history)
        return np.stack([past[-1 - i * self.sample_interval_steps]
                         for i in range(self.frames - 1, -1, -1)], axis=0)

    @staticmethod
    def _validate(frame_rgb: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame_rgb)
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError(f"Expected HxWx3 uint8 RGB image, got {frame.shape} {frame.dtype}")
        return np.ascontiguousarray(frame)


class PurePursuitController:
    """Convert predicted forward/right waypoints into MetaDrive actions."""

    def __init__(
        self,
        target_speed_kmh: float = 20.0,
        wheelbase_m: float = 2.7,
        max_steering_deg: float = 60.0,
        steering_gain: float = 1.0,
        lookahead_m: float = 4.0,
        speed_gain: float = 0.35,
    ):
        if target_speed_kmh <= 0 or wheelbase_m <= 0 or max_steering_deg <= 0:
            raise ValueError("speed, wheelbase, and maximum steering must be positive")
        self.target_speed_mps = target_speed_kmh / 3.6
        self.wheelbase_m = wheelbase_m
        self.max_steering_rad = np.deg2rad(max_steering_deg)
        self.steering_gain = steering_gain
        self.lookahead_m = lookahead_m
        self.speed_gain = speed_gain
        self.last_steering = 0.0

    def reset(self) -> None:
        self.last_steering = 0.0

    def action(self, trajectory: np.ndarray, speed_mps: float) -> tuple[float, float, dict]:
        points = np.asarray(trajectory, dtype=np.float64)
        if points.shape != (20, 2) or not np.isfinite(points).all():
            raise ValueError(f"Expected finite (20, 2) trajectory, got {points.shape}")
        speed_mps = max(0.0, float(speed_mps))

        # Select the first point beyond a speed-dependent lookahead. The
        # learned horizon is 5 s, so this stays within the model's forecast.
        desired_lookahead = max(self.lookahead_m, 1.5 + 0.45 * speed_mps)
        forward = points[:, 0]
        valid = np.flatnonzero(forward > 0.5)
        if len(valid) == 0:
            target = np.array([1.0, 0.0])
        else:
            candidates = valid[forward[valid] >= desired_lookahead]
            index = int(candidates[0] if len(candidates) else valid[-1])
            target = points[index]

        x, y = float(target[0]), float(target[1])
        distance_sq = max(x * x + y * y, 1e-4)
        curvature = 2.0 * y / distance_sq
        # Waypoint y is positive to the vehicle's right, while MetaDrive's
        # positive steering rotates heading counter-clockwise (left).
        steering_angle = -np.arctan(self.wheelbase_m * curvature)
        steering = float(np.clip(
            self.steering_gain * steering_angle / self.max_steering_rad,
            -1.0,
            1.0,
        ))
        self.last_steering = steering

        # The trajectory is a path, not a speed command. Following its
        # forecast speed creates a stationary fixed point at startup: the
        # constant-velocity model predicts zero distance when speed is zero.
        target_speed = self.target_speed_mps
        if abs(steering) > 0.65:
            target_speed *= 0.65
        acceleration = float(np.clip(
            self.speed_gain * (target_speed - speed_mps),
            -1.0,
            1.0,
        ))
        diagnostics = {
            "target_x_m": x,
            "target_y_m": y,
            "curvature": float(curvature),
            "target_speed_mps": float(target_speed),
        }
        return steering, acceleration, diagnostics


class ModelTrajectoryPolicy:
    """Run inference at the training cadence and follow its predicted path."""

    def __init__(self, checkpoint: Path, device: str = "auto", history_interval: int = 5, **controller_kwargs):
        self.model = ResidualTrajectoryModel(checkpoint, device=device)
        self.history = TemporalFrameHistory(4, history_interval)
        self.controller = PurePursuitController(**controller_kwargs)
        self.latest_trajectory = None
        self.last_inference_step = None
        self.inference_count = 0

    def reset(self, frame_rgb: np.ndarray) -> None:
        self.history.reset(frame_rgb)
        self.controller.reset()
        self.latest_trajectory = None
        self.last_inference_step = None
        self.inference_count = 0

    def act(self, frame_rgb: np.ndarray, speed_mps: float, step: int) -> tuple[float, float, dict]:
        if not self.history.ready:
            self.reset(frame_rgb)
        elif self.history.update(frame_rgb, step):
            self.latest_trajectory = self.model.predict(self.history.as_array(), speed_mps)
            self.last_inference_step = step
            self.inference_count += 1
        if self.latest_trajectory is None:
            self.latest_trajectory = self.model.predict(self.history.as_array(), speed_mps)
            self.last_inference_step = step
            self.inference_count += 1
        steering, acceleration, diagnostics = self.controller.action(self.latest_trajectory, speed_mps)
        diagnostics.update({
            "inference_step": self.last_inference_step,
            "model_device": str(self.model.device),
        })
        return steering, acceleration, diagnostics


class ZeroSteeringPolicy:
    """Simple reproducible baseline using the same longitudinal servo."""

    def __init__(self, target_speed_kmh: float = 20.0):
        self.target_speed_mps = target_speed_kmh / 3.6

    def reset(self, frame_rgb: np.ndarray) -> None:
        del frame_rgb

    def act(self, frame_rgb: np.ndarray, speed_mps: float, step: int) -> tuple[float, float, dict]:
        del frame_rgb, step
        acceleration = float(np.clip(0.35 * (self.target_speed_mps - max(0.0, speed_mps)), -1.0, 1.0))
        return 0.0, acceleration, {"inference_step": None}
