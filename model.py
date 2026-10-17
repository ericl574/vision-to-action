
import torch
import torch.nn as nn


class DrivingModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )

        self.regressor = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, images):
        features = self.features(images)
        return self.regressor(features)


if __name__ == "__main__":
    model = DrivingModel()

    # Four RGB images, each 128 x 128 pixels
    images = torch.randn(4, 3, 128, 128)

    predictions = model(images)

    assert predictions.shape == (4, 1)
    assert torch.isfinite(predictions).all()

    print("Model initialized successfully!")
    print("Predictions shape:", predictions.shape)
    print("Trainable parameters:",
          sum(p.numel() for p in model.parameters()))

