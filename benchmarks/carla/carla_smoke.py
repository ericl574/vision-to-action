"""Box smoke test: start CARLA headless, render 30 camera frames, save one PNG.

    source /workspace/carla_env.sh
    python benchmarks/carla/carla_smoke.py --out /workspace/smoke_camera.png

Prints ``SMOKE OK`` with the achieved simulation rate on success.
"""

import argparse
import queue
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from server import kill_server, launch_server, wait_for_server  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--carla-root", default=None)
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default="Town10HD_Opt")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--out", default="smoke_camera.png")
    parser.add_argument("--no-launch", action="store_true", help="connect to an already running server")
    args = parser.parse_args()

    import carla
    import cv2

    proc = None if args.no_launch else launch_server(args.carla_root, args.port, log_path="carla_server_smoke.log")
    actors = []
    try:
        client = wait_for_server(port=args.port)
        print("server version", client.get_server_version())
        world = client.get_world()
        if args.town not in world.get_map().name:
            world = client.load_world(args.town)
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1
        world.apply_settings(settings)

        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.find("vehicle.tesla.model3")
        spawn = world.get_map().get_spawn_points()[0]
        vehicle = world.spawn_actor(vehicle_bp, spawn)
        actors.append(vehicle)
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", "1024")
        cam_bp.set_attribute("image_size_y", "576")
        cam_bp.set_attribute("fov", "100")
        camera = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.3, z=1.7)), attach_to=vehicle)
        actors.append(camera)
        images: queue.Queue = queue.Queue()
        camera.listen(images.put)

        vehicle.apply_control(carla.VehicleControl(throttle=0.4))
        start = time.time()
        image = None
        for _ in range(args.frames):
            world.tick()
            image = images.get(timeout=10.0)
        elapsed = time.time() - start
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)[:, :, :3]
        cv2.imwrite(args.out, array)
        print(f"SMOKE OK frames={args.frames} sim_fps={args.frames / elapsed:.1f} saved={args.out}")
    finally:
        try:
            for actor in reversed(actors):
                if hasattr(actor, "stop"):
                    actor.stop()
                actor.destroy()
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
        except Exception:  # noqa: BLE001
            pass
        kill_server(proc, args.port if proc is not None else None)


if __name__ == "__main__":
    main()
