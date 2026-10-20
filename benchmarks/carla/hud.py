"""Offline FSD-style overlay renderer.

Every widget is driven by the network's own outputs logged during the run (steer, desired
speed, traffic-light probabilities, hazard probability/distance). When a log carries only
ground-truth labels (an expert run), the panel says so in its header. Pure numpy/OpenCV.

    python benchmarks/carla/hud.py render --log /workspace/runs/eval_v1/Town10HD_Opt_dev_003 --out clip.mp4
    python benchmarks/carla/hud.py reel --spec reel.json --out demo/pilot_reel.mp4
"""

import argparse
import csv
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

CANVAS_W = 1024
CAM_H = 576
PANEL_H = 176
BG = (24, 20, 16)          # BGR dark panel
FG = (235, 235, 235)
DIM = (120, 120, 120)
ACCENT = (255, 190, 60)    # BGR: blue-ish amber swap -> looks orange/blue in Tesla style
GOOD = (90, 200, 90)
WARN = (60, 200, 255)
BAD = (70, 70, 230)
TL_COLORS = [(110, 110, 110), (90, 200, 90), (60, 200, 255), (70, 70, 230)]
TL_LABELS = ["none", "green", "yellow", "red"]
COMMAND_LABELS = ["FOLLOW LANE", "TURN LEFT", "TURN RIGHT", "STRAIGHT"]


def _text(img, text, org, scale=0.55, color=FG, thickness=1):
    import cv2

    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _bar(img, x, y, w, h, frac, color):
    import cv2

    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), 1)
    fill = int(round(max(0.0, min(1.0, frac)) * (w - 2)))
    if fill > 0:
        cv2.rectangle(img, (x + 1, y + 1), (x + 1 + fill, y + h - 1), color, -1)


def decision_caption(target_mps: float, speed_mps: float, tl_probs, hazard_prob: float, source: str = "network sees") -> str:
    """Plain-language caption derived only from the panel's own values (network outputs, or labels on expert runs)."""
    tl = int(np.argmax(tl_probs)) if tl_probs is not None else 0
    if target_mps < 0.5:
        if tl == 3 and tl_probs[3] > 0.5:
            return f"STOPPING: {source} red light"
        if hazard_prob > 0.5:
            return f"STOPPING: {source} hazard ahead"
        return "STOPPING"
    if target_mps < speed_mps - 1.0:
        return "SLOWING: desired %.0f km/h" % (target_mps * 3.6)
    return "CRUISING: desired %.0f km/h" % (target_mps * 3.6)


def render_frame(rgb: np.ndarray, out: dict, name: str = "PILOT", ground_truth: bool = False) -> np.ndarray:
    """Compose camera frame + panel. ``out`` keys: steer, target_speed_mps, speed_mps, command,
    tl_probs (4), hazard_prob, hazard_dist_m, t. Returns a BGR image (CAM_H+PANEL_H, CANVAS_W, 3)."""
    import cv2

    cam = cv2.resize(rgb[:, :, ::-1], (CANVAS_W, CAM_H), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((CAM_H + PANEL_H, CANVAS_W, 3), dtype=np.uint8)
    canvas[:CAM_H] = cam
    canvas[CAM_H:] = BG

    header = f"{name}  |  end-to-end  |  camera only" if not ground_truth else f"EXPERT RUN  |  ground-truth labels shown"
    cv2.rectangle(canvas, (0, 0), (CANVAS_W, 30), (0, 0, 0), -1)
    _text(canvas, header, (12, 21), 0.55, FG if not ground_truth else WARN)
    _text(canvas, f"t = {float(out.get('t', 0.0)):5.1f} s", (CANVAS_W - 130, 21), 0.55, DIM)

    y0 = CAM_H
    steer = float(out.get("steer", 0.0))
    speed = float(out.get("speed_mps", 0.0))
    target = float(out.get("target_speed_mps", 0.0))
    command = int(out.get("command", 0))
    tl_probs = out.get("tl_probs") or [1.0, 0.0, 0.0, 0.0]
    hazard_prob = float(out.get("hazard_prob", 0.0))
    hazard_dist = float(out.get("hazard_dist_m", 0.0))

    # Steering gauge
    cx, cy, radius = 90, y0 + 100, 60
    cv2.ellipse(canvas, (cx, cy), (radius, radius), 0, 180, 360, (70, 70, 70), 3, cv2.LINE_AA)
    angle = np.deg2rad(270 + steer * 80)
    tip = (int(cx + radius * np.cos(angle)), int(cy + radius * np.sin(angle)))
    cv2.line(canvas, (cx, cy), tip, ACCENT, 3, cv2.LINE_AA)
    _text(canvas, "STEER", (cx - 24, y0 + 22), 0.5, DIM)
    _text(canvas, f"{steer:+.2f}", (cx - 28, cy + 30), 0.6, FG)

    # Speed vs desired
    x = 200
    _text(canvas, "SPEED", (x, y0 + 22), 0.5, DIM)
    _text(canvas, f"{speed * 3.6:4.0f}", (x - 4, y0 + 78), 1.5, FG, 2)
    _text(canvas, "km/h", (x + 92, y0 + 78), 0.5, DIM)
    _text(canvas, f"desired {target * 3.6:4.0f} km/h", (x, y0 + 108), 0.55, ACCENT)
    _bar(canvas, x, y0 + 120, 150, 10, speed / 14.0, FG)
    _bar(canvas, x, y0 + 136, 150, 10, target / 14.0, ACCENT)
    _text(canvas, "actual / desired", (x, y0 + 162), 0.42, DIM)

    # Navigation command
    x = 400
    _text(canvas, "ROUTE", (x, y0 + 22), 0.5, DIM)
    arrows = {0: "^", 1: "<-", 2: "->", 3: "^^"}
    _text(canvas, arrows.get(command, "^"), (x + 10, y0 + 85), 1.6, FG, 3)
    _text(canvas, COMMAND_LABELS[command] if 0 <= command < 4 else "?", (x, y0 + 115), 0.5, FG)
    _text(canvas, "(navigation input)", (x, y0 + 140), 0.42, DIM)

    # Network perception readouts
    x = 580
    _text(canvas, "NETWORK SEES" if not ground_truth else "GROUND TRUTH", (x, y0 + 22), 0.5, DIM)
    _text(canvas, "traffic light", (x, y0 + 42), 0.45, DIM)
    for i in range(4):
        yy = y0 + 50 + i * 20
        _text(canvas, TL_LABELS[i], (x, yy + 11), 0.42, FG)
        _bar(canvas, x + 60, yy + 2, 110, 11, float(tl_probs[i]), TL_COLORS[i])
        _text(canvas, f"{float(tl_probs[i]):.2f}", (x + 176, yy + 11), 0.4, DIM)
    _text(canvas, "hazard ahead", (x, y0 + 146), 0.45, DIM)
    _bar(canvas, x + 100, y0 + 138, 70, 11, hazard_prob, BAD if hazard_prob > 0.5 else DIM)
    _text(canvas, f"{hazard_prob:.2f}  {hazard_dist:4.0f} m" if hazard_prob > 0.5 else f"{hazard_prob:.2f}",
          (x + 176, y0 + 147), 0.42, FG)

    # Decision caption
    x = 820
    _text(canvas, "DECISION", (x, y0 + 22), 0.5, DIM)
    caption = decision_caption(target, speed, tl_probs, hazard_prob, "label:" if ground_truth else "network sees")
    words = caption.split(": ")
    color = BAD if caption.startswith("STOPPING") else (WARN if caption.startswith("SLOWING") else GOOD)
    _text(canvas, words[0], (x, y0 + 60), 0.75, color, 2)
    if len(words) > 1:
        for j, line in enumerate(_wrap(words[1], 22)):
            _text(canvas, line, (x, y0 + 90 + 22 * j), 0.5, FG)
    return canvas


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width and current:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def load_log(log_dir) -> tuple[list[dict], bool]:
    """Read ``outputs.csv`` (network run) or ``labels.csv`` (expert run) into HUD rows."""
    log_dir = Path(log_dir)
    path = log_dir / "outputs.csv" if (log_dir / "outputs.csv").exists() else log_dir / "labels.csv"
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    network = "tl_red" in rows[0] if rows else False
    hud_rows = []
    for row in rows:
        if network:
            tl = [float(row["tl_none"]), float(row["tl_green"]), float(row["tl_yellow"]), float(row["tl_red"])]
            hazard_prob = float(row["hazard_prob"])
            hazard_dist = float(row["hazard_dist_m"] or 0)
        else:
            state = int(float(row["tl_state"]))
            tl = [1.0 if i == state else 0.0 for i in range(4)]
            hazard_prob = float(row["hazard"])
            hazard_dist = float(row["hazard_dist_m"] or 0)
        hud_rows.append({
            "idx": int(row["idx"]), "t": float(row["t"]), "speed_mps": float(row["speed_mps"]),
            "command": int(float(row["command"])), "steer": float(row["steer"]),
            "target_speed_mps": float(row["target_speed_mps"]), "tl_probs": tl,
            "hazard_prob": hazard_prob, "hazard_dist_m": hazard_dist,
        })
    return hud_rows, not network


def _transcode(src: Path, dst: Path) -> None:
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "23", "-movflags", "+faststart", str(dst)], check=True)
        src.unlink(missing_ok=True)
    else:
        shutil.move(str(src), str(dst))


def render_video(log_dir, out_mp4, fps: int = 10, name: str = "PILOT", start_s: float = 0.0, end_s: float | None = None,
                 caption: str | None = None) -> Path:
    import cv2

    rows, ground_truth = load_log(log_dir)
    frames_dir = Path(log_dir) / "frames"
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_mp4.with_suffix(".tmp.mp4")
    writer = None
    for row in rows:
        if row["t"] < start_s or (end_s is not None and row["t"] > end_s):
            continue
        bgr = cv2.imread(str(frames_dir / f"{row['idx']:06d}.jpg"))
        if bgr is None:
            continue
        frame = render_frame(bgr[:, :, ::-1], row, name=name, ground_truth=ground_truth)
        if caption:
            cv2.rectangle(frame, (0, 30), (CANVAS_W, 62), (0, 0, 0), -1)
            _text(frame, caption, (12, 53), 0.6, ACCENT)
        if writer is None:
            writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1], frame.shape[0]))
        writer.write(frame)
    if writer is None:
        raise RuntimeError(f"No frames rendered from {log_dir}")
    writer.release()
    _transcode(tmp, out_mp4)
    return out_mp4


def make_reel(clips: list[dict], out_mp4, fps: int = 10, name: str = "PILOT") -> Path:
    """``clips``: [{"log": dir, "start": s, "end": s, "caption": str}, ...] -> one concatenated MP4."""
    out_mp4 = Path(out_mp4)
    with tempfile.TemporaryDirectory() as tmpdir:
        parts = []
        for i, clip in enumerate(clips):
            part = Path(tmpdir) / f"part_{i:02d}.mp4"
            render_video(clip["log"], part, fps=fps, name=name, start_s=clip.get("start", 0.0),
                         end_s=clip.get("end"), caption=clip.get("caption"))
            parts.append(part)
        listing = Path(tmpdir) / "list.txt"
        listing.write_text("".join(f"file '{p}'\n" for p in parts))
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
                        "-c", "copy", str(out_mp4)], check=True)
    return out_mp4


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    render = sub.add_parser("render")
    render.add_argument("--log", required=True)
    render.add_argument("--out", required=True)
    render.add_argument("--fps", type=int, default=10)
    render.add_argument("--name", default="PILOT")
    render.add_argument("--start", type=float, default=0.0)
    render.add_argument("--end", type=float, default=None)
    render.add_argument("--caption", default=None)
    reel = sub.add_parser("reel")
    reel.add_argument("--spec", required=True, help="JSON list of {log, start, end, caption}")
    reel.add_argument("--out", required=True)
    reel.add_argument("--fps", type=int, default=10)
    reel.add_argument("--name", default="PILOT")
    args = parser.parse_args()
    if args.cmd == "render":
        print(render_video(args.log, args.out, args.fps, args.name, args.start, args.end, args.caption))
    else:
        with open(args.spec) as handle:
            clips = json.load(handle)
        print(make_reel(clips, args.out, args.fps, args.name))


if __name__ == "__main__":
    main()
