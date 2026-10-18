
import base64
from io import BytesIO
from pathlib import Path

import eventlet
import socketio
import torch
from PIL import Image
from torchvision import transforms

from model import DrivingModel


# -------- Configuration --------

DRIVE_ENABLED = True  # Enable after inference test
TARGET_SPEED = 30.0
PORT = 4567

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

# Must match dataset.py training preprocessing
preprocess = transforms.Compose([
    transforms.Resize((96, 192)),
    transforms.ToTensor(),
])


# -------- Load Trained Model --------

checkpoint_path = (
    Path(__file__).resolve().parent / "checkpoints" / "best.pt"
)

model = DrivingModel().to(DEVICE)

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True,
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"Model loaded on {DEVICE}")
print(f"Checkpoint epoch: {checkpoint['epoch']}")


# -------- Socket.IO Server --------

sio = socketio.Server(async_mode="eventlet")
app = socketio.WSGIApp(sio)

frame_count = 0


def send_control(sid, steering, throttle):
    sio.emit(
        "steer",
        data={
            "steering_angle": str(steering),
            "throttle": str(throttle),
        },
        room=sid,
    )


@sio.on("connect")
def connect(sid, environ):
    print("Simulator connected!")
    send_control(sid, 0.0, 0.0)


@sio.on("telemetry")
def telemetry(sid, data):
    global frame_count

    if not data or not data.get("image"):
        send_control(sid, 0.0, 0.0)
        return

    try:
        speed = float(data.get("speed", 0) or 0)

        # Decode the simulator camera image
        raw_image = base64.b64decode(
            data["image"].split(",")[-1]
        )

        with Image.open(BytesIO(raw_image)) as image:
            image_tensor = preprocess(
                image.convert("RGB")
            )

        image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

        # Model inference
        with torch.inference_mode():
            steering = model(image_tensor).item()

        steering = max(-1.0, min(1.0, steering))

        # Conservative speed for the first simulation test
        throttle = 0.18 if speed < TARGET_SPEED else 0.0

        frame_count += 1

        if frame_count % 30 == 0:
            print(
                f"Frame {frame_count} | "
                f"Speed: {speed:.2f} | "
                f"AI Steering: {steering:.4f}"
            )

        if DRIVE_ENABLED:
            send_control(sid, steering, throttle)
        else:
            send_control(sid, 0.0, 0.0)

    except Exception as error:
        print("Inference error:", error)
        send_control(sid, 0.0, 0.0)


@sio.on("disconnect")
def disconnect(sid):
    print("Simulator disconnected")


if __name__ == "__main__":
    print("Waiting for Simulator on port 4567...")
    print("Driving enabled:", DRIVE_ENABLED)

    eventlet.wsgi.server(
        eventlet.listen(("127.0.0.1", PORT)),
        app,
        log_output=False,
    )
