"""Camera-history behavioral-cloning steering model and runtime policy."""

from pathlib import Path

import numpy as np

from benchmarks.metadrive.model_controller import TemporalFrameHistory


def create_network():
    import torch
    from torch import nn
    from train_comma1m import CNNGRU

    class ActionCNNGRU(nn.Module):
        def __init__(self):
            super().__init__()
            backbone = CNNGRU()
            self.encoder = backbone.encoder
            self.gru = backbone.gru
            self.head = nn.Sequential(nn.Linear(129, 128), nn.ReLU(), nn.Linear(128, 1), nn.Tanh())

        def forward(self, images, speed):
            batch, time, channels, height, width = images.shape
            encoded = self.encoder(images.reshape(batch * time, channels, height, width))
            encoded = encoded.reshape(batch, time, -1)
            encoded, _ = self.gru(encoded)
            return self.head(torch.cat((encoded[:, -1], speed), dim=1)).squeeze(-1)

    return ActionCNNGRU()


class ActionPolicy:
    def __init__(self, checkpoint: Path, device: str = "auto", history_interval: int = 5,
                 target_speed_kmh: float = 20.0):
        import torch

        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = create_network().to(self.device)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()
        self.checkpoint = Path(checkpoint)
        self.epoch = payload.get("epoch")
        self.validation_mae = payload.get("val_mae")
        self.history = TemporalFrameHistory(4, history_interval)
        self.target_speed_mps = target_speed_kmh / 3.6
        self.latest_trajectory = None
        self.last_inference_step = None
        self.inference_count = 0
        self.steering = 0.0

    def reset(self, frame_rgb: np.ndarray):
        self.history.reset(frame_rgb)
        self.last_inference_step = None
        self.inference_count = 0
        self.steering = 0.0

    def act(self, frame_rgb: np.ndarray, speed_mps: float, step: int):
        import torch

        if not self.history.ready:
            self.history.reset(frame_rgb)
        if self.history.update(frame_rgb, step) or self.last_inference_step is None:
            frames = self.history.as_array()
            images = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div_(255.0)
            images = images.unsqueeze(0).to(self.device)
            speed = torch.tensor([[speed_mps / 30.0]], dtype=torch.float32, device=self.device)
            with torch.inference_mode():
                self.steering = float(self.model(images, speed)[0].cpu())
            self.last_inference_step = step
            self.inference_count += 1
        acceleration = float(np.clip(0.35 * (self.target_speed_mps - speed_mps), -1.0, 1.0))
        return self.steering, acceleration, {"inference_step": self.last_inference_step}
