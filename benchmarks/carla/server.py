"""Launch and connect to a headless CARLA server.

The server binary refuses to run as root, so when this process is root the server is
started as the ``carla`` user created by ``setup_box.sh``.
"""

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path


def launch_server(carla_root: str | None = None, port: int = 2000, quality: str = "Epic",
                  gpu: int = 0, log_path: str | None = None, user: str = "carla") -> subprocess.Popen:
    carla_root = carla_root or os.environ.get("CARLA_ROOT")
    if not carla_root:
        raise RuntimeError("Set CARLA_ROOT or pass --carla-root")
    script = Path(carla_root) / "CarlaUE4.sh"
    if not script.exists():
        raise FileNotFoundError(script)
    cmd = ["bash", str(script), "-RenderOffScreen", "-nosound", f"-carla-rpc-port={port}",
           f"-quality-level={quality}", f"-graphicsadapter={gpu}"]
    if os.geteuid() == 0 and shutil.which("runuser"):
        cmd = ["runuser", "-u", user, "--"] + cmd
    log = open(log_path, "ab") if log_path else subprocess.DEVNULL
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, preexec_fn=os.setsid,
                            cwd=str(carla_root))


def wait_for_server(host: str = "127.0.0.1", port: int = 2000, timeout: float = 180.0):
    """Return a connected ``carla.Client`` once the RPC port answers."""
    import carla

    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        try:
            client = carla.Client(host, port)
            client.set_timeout(10.0)
            client.get_server_version()
            client.set_timeout(120.0)
            return client
        except Exception as error:  # noqa: BLE001 - RPC not up yet
            last_error = error
            time.sleep(2.0)
    raise TimeoutError(f"CARLA server on {host}:{port} did not answer within {timeout}s: {last_error}")


def kill_server(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=15)
    except Exception:  # noqa: BLE001
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:  # noqa: BLE001
            pass
