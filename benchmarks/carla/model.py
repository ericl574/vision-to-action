"""End-to-end camera driving network (``PilotNet``) and its runtime policy.

The network maps a short history of front-camera frames + ego speed + a navigation command to
steering and desired speed. The desired speed is turned into pedal commands by ``SpeedServo``,
a fixed proportional loop that reads nothing but the network output and the speedometer.
Auxiliary heads (traffic-light state, hazard ahead, hazard distance) are training signal and
overlay content only; they never touch control.
"""

from pathlib import Path

import numpy as np

SPEED_NORM_MPS = 10.0
DIST_NORM_M = 30.0
TL_NAMES = ("none", "green", "yellow", "red")


def create_network(history: int = 3, pretrained: bool = True, backbone: str = "resnet34",
                   gru_hidden: int = 256, speed_dropout: float = 0.3, n_commands: int = 4):
    import torch
    import torchvision
    from torch import nn
    from torch.nn import functional as F

    class PilotNet(nn.Module):
        def __init__(self):
            super().__init__()
            weights = None
            if pretrained:
                weights = {"resnet18": torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
                           "resnet34": torchvision.models.ResNet34_Weights.IMAGENET1K_V1}[backbone]
            trunk = getattr(torchvision.models, backbone)(weights=weights)
            feat = trunk.fc.in_features
            trunk.fc = nn.Identity()
            self.encoder = trunk
            self.gru = nn.GRU(feat, gru_hidden, batch_first=True)
            self.speed_mlp = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
            self.speed_dropout = speed_dropout
            self.branches = nn.ModuleList([
                nn.Sequential(nn.Linear(gru_hidden + 64, 256), nn.ReLU(), nn.Dropout(0.2),
                              nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 2))
                for _ in range(n_commands)
            ])
            self.tl_head = nn.Sequential(nn.Linear(gru_hidden, 128), nn.ReLU(), nn.Linear(128, 4))
            self.hazard_head = nn.Sequential(nn.Linear(gru_hidden, 128), nn.ReLU(), nn.Linear(128, 2))
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            self.config = dict(history=history, backbone=backbone, gru_hidden=gru_hidden,
                               speed_dropout=speed_dropout, n_commands=n_commands)

        def forward(self, images, speed_mps, command):
            batch, time, channels, height, width = images.shape
            flat = (images.reshape(batch * time, channels, height, width) - self.mean) / self.std
            encoded = self.encoder(flat).reshape(batch, time, -1)
            _, hidden = self.gru(encoded)
            features = hidden[-1]
            speed = speed_mps / SPEED_NORM_MPS
            if self.training and self.speed_dropout > 0:
                keep = (torch.rand(batch, 1, device=speed.device) > self.speed_dropout).float()
                speed = speed * keep
            fused = torch.cat((features, self.speed_mlp(speed)), dim=1)
            stacked = torch.stack([branch(fused) for branch in self.branches], dim=1)  # [B, C, 2]
            chosen = stacked[torch.arange(batch, device=images.device), command]
            hazard = self.hazard_head(features)
            return {
                "steer": torch.tanh(chosen[:, 0]),
                "target_speed": F.softplus(chosen[:, 1]) * SPEED_NORM_MPS,
                "tl_logits": self.tl_head(features),
                "hazard_logit": hazard[:, 0],
                "hazard_dist": F.softplus(hazard[:, 1]) * DIST_NORM_M,
            }

    return PilotNet()


class SpeedServo:
    """Desired speed -> (throttle, brake). Proportional, with a hold-brake at standstill."""

    def __init__(self, kp: float = 0.5, max_throttle: float = 0.7, deadband_mps: float = 0.3):
        self.kp = kp
        self.max_throttle = max_throttle
        self.deadband = deadband_mps

    def __call__(self, target_mps: float, speed_mps: float) -> tuple[float, float]:
        if target_mps < 0.3 and speed_mps < 0.5:
            return 0.0, 1.0
        error = target_mps - speed_mps
        if error >= 0:
            return float(min(self.max_throttle, self.kp * error)), 0.0
        return 0.0, float(np.clip(-self.kp * error - self.deadband, 0.0, 1.0))


class FrameHistory:
    """Keeps the last ``history`` frames sampled every ``stride`` steps; pads with the first frame."""

    def __init__(self, history: int = 3, stride: int = 3):
        self.history = history
        self.stride = stride
        self.buffer: list[np.ndarray] = []

    def reset(self, frame: np.ndarray) -> None:
        self.buffer = [frame] * ((self.history - 1) * self.stride + 1)

    def push(self, frame: np.ndarray) -> None:
        if not self.buffer:
            self.reset(frame)
            return
        self.buffer.append(frame)
        del self.buffer[0]

    def stack(self) -> np.ndarray:
        indices = [len(self.buffer) - 1 - k * self.stride for k in reversed(range(self.history))]
        return np.stack([self.buffer[i] for i in indices])


class PilotPolicy:
    """Runtime wrapper: RGB frame + speed + command in, pedals out. No privileged inputs."""

    def __init__(self, checkpoint, device: str = "auto"):
        import torch

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = payload["config"]
        self.config = cfg
        self.width, self.height = int(cfg["width"]), int(cfg["height"])
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = torch.device(device)
        self.model = create_network(history=cfg["history"], pretrained=False, backbone=cfg.get("backbone", "resnet34"),
                                    gru_hidden=cfg.get("gru_hidden", 256), speed_dropout=0.0).to(self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()
        self.history = FrameHistory(cfg["history"], cfg.get("stride", 3))
        self.servo = SpeedServo()
        self.checkpoint = Path(checkpoint)
        self.epoch = payload.get("epoch")
        self.last_outputs: dict = {}

    def prepare(self, frame_rgb: np.ndarray) -> np.ndarray:
        if frame_rgb.shape[0] != self.height or frame_rgb.shape[1] != self.width:
            import cv2

            frame_rgb = cv2.resize(frame_rgb, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(frame_rgb)

    def reset(self, frame_rgb: np.ndarray) -> None:
        self.history.reset(self.prepare(frame_rgb))
        self.last_outputs = {}

    def act(self, frame_rgb: np.ndarray, speed_mps: float, command: int) -> tuple[float, float, float, dict]:
        import torch

        self.history.push(self.prepare(frame_rgb))
        frames = torch.from_numpy(self.history.stack()).permute(0, 3, 1, 2).float().div_(255.0).unsqueeze(0)
        speed = torch.tensor([[speed_mps]], dtype=torch.float32)
        cmd = torch.tensor([int(command)], dtype=torch.long)
        with torch.inference_mode():
            out = self.model(frames.to(self.device), speed.to(self.device), cmd.to(self.device))
            tl_probs = torch.softmax(out["tl_logits"][0], dim=0).cpu().numpy()
            outputs = {
                "steer": float(out["steer"][0]),
                "target_speed_mps": float(out["target_speed"][0]),
                "tl_probs": tl_probs.tolist(),
                "hazard_prob": float(torch.sigmoid(out["hazard_logit"][0])),
                "hazard_dist_m": float(out["hazard_dist"][0]),
            }
        throttle, brake = self.servo(outputs["target_speed_mps"], speed_mps)
        self.last_outputs = outputs
        return outputs["steer"], throttle, brake, outputs
