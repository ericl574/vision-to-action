"""Build an initial US-only camera-to-future-trajectory dataset from comma1M.

Frame alignment: frame_t/frame_states are at ~20Hz. Raw HEVC fps metadata is ignored.
The local xy axes use current estimated travel direction as forward and a
horizontal right axis; this is NOT an IMU/device-heading transformation.
"""
import argparse, csv, json
from pathlib import Path
import cv2
import numpy as np
from safetensors.numpy import load_file
from pymap3d import ecef2geodetic

# Country selection is a coarse US geographic envelope. User's first ten
# segments have been manually reviewed based on reported coordinates.
# One explicit foreign segment is excluded. For larger batches, review GPS
# coordinates rather than relying solely on the envelope.
EXCLUDED = {'002c388a95dba92f8a990e0cf5d259ea'}

def xy_trajectory(states, index, future_indices):
    pos = states[:, :3]
    vel = states[index, 7:10]
    p0 = pos[index]
    # Earth radial direction = local up, adequate for short 5s paths.
    up = p0 / max(np.linalg.norm(p0), 1.0)
    forward = vel - up * np.dot(vel, up)
    speed = np.linalg.norm(forward)
    if speed < 1.5:
        return None  # heading from velocity unreliable when nearly stationary
    forward /= speed
    right = np.cross(forward, up)
    right /= max(np.linalg.norm(right), 1e-9)
    delta = pos[future_indices] - p0
    return np.stack((delta @ forward, delta @ right), axis=-1).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='data/external/comma1m/data')
    ap.add_argument('--output', default='data/processed/comma1m_v1')
    ap.add_argument('--stride', type=int, default=5, help='Training examples every N camera frames (20Hz).')
    ap.add_argument('--max-segments', type=int, default=0)
    ap.add_argument('--segment-id', default=None)
    args = ap.parse_args()
    if args.stride < 1: ap.error('stride must be positive')
    source, output = Path(args.source), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    segments = sorted(source.glob('*/localizer.safetensors'))
    if args.segment_id:
        segments = [p for p in segments if p.parent.name == args.segment_id]
    if args.max_segments: segments = segments[:args.max_segments]
    csv_path = output / 'samples.csv'
    summary_path = output / 'summary.json'
    if csv_path.exists() != summary_path.exists():
        raise RuntimeError('Existing samples.csv and summary.json are incomplete; stopping safely')
    if csv_path.exists():
        with csv_path.open(newline='') as f:
            rows = list(csv.DictReader(f))
        summary = json.loads(summary_path.read_text())
    else:
        rows, summary = [], []
    completed = {s['segment'] for s in summary}
    for loc_path in segments:
        sid = loc_path.parent.name
        if sid in completed:
            print('SKIP already processed', sid, flush=True)
            continue
        if sid in EXCLUDED:
            print('SKIP explicitly non-US segment', sid, flush=True)
            continue
        video = loc_path.parent / 'fcamera.hevc'
        if not video.exists():
            print('SKIP missing video', sid, flush=True); continue
        d = load_file(str(loc_path))
        ts = np.asarray(d['frame_t'], dtype=np.float64).reshape(-1)
        states = np.asarray(d['frame_states'], dtype=np.float64)
        if len(ts)!=len(states) or len(ts)<150 or not np.all(np.diff(ts)>0):
            print('SKIP invalid timestamps', sid, flush=True); continue
        dt = np.diff(ts)
        if not (0.045 < np.median(dt) < 0.055):
            print('SKIP unusual timestamp cadence', sid, flush=True); continue
        lat, lon, _ = ecef2geodetic(*states[:, :3].T)
        latitude, longitude = float(np.median(lat)), float(np.median(lon))
        # Geographic inclusion intentionally conservative for mainland US.
        if not (24.0 <= latitude <= 49.5 and -125.0 <= longitude <= -66.0):
            print('SKIP outside continental-US bounds', sid, latitude, longitude, flush=True); continue
        # Use timestamps, not frame-index arithmetic, for future targets.
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened(): print('SKIP unable to open video', sid); continue
        dest = output / sid
        dest.mkdir(parents=True, exist_ok=True)
        decoded = 0
        # Read every frame, but only persist those necessary for sample histories.
        while True:
            ok, frame = cap.read()
            if not ok: break
            if decoded % args.stride == 0:
                resized = cv2.resize(frame, (192, 120), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(dest / f'{decoded:04d}.jpg'), resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
            decoded += 1
        cap.release()
        if abs(decoded-len(ts))>2:
            print(f'SKIP frame-count mismatch {sid}: decoded={decoded}, timestamps={len(ts)}', flush=True)
            # Images may remain for inspection, but never get training rows.
            continue
        n = min(decoded, len(ts))
        accepted = 0
        for i in range(30, n-102, args.stride):
            history = [i-30, i-20, i-10, i]
            if any(h % args.stride != 0 for h in history): continue
            # Target times: 0.25, 0.50, ..., 5.00 seconds
            target_time = ts[i] + 0.25*np.arange(1, 21)
            indices = np.searchsorted(ts[:n], target_time)
            if np.any(indices>=n) or np.any(indices<=0): continue
            before = indices-1
            nearest = np.where(abs(ts[indices]-target_time)<abs(ts[before]-target_time), indices, before)
            if np.max(abs(ts[nearest]-target_time))>0.030: continue
            traj = xy_trajectory(states, i, nearest)
            if traj is None or not np.isfinite(traj).all(): continue
            # Filter erratic localization jumps and physically implausible 5s motion.
            positions = states[np.r_[i, nearest], :3]
            step = np.linalg.norm(np.diff(positions, axis=0), axis=1)
            if step.max()>20 or np.linalg.norm(traj[-1])>220: continue
            speed = float(np.linalg.norm(states[i, 7:10]))
            if speed > 60: continue
            rows.append({'segment':sid,'index':i,'history':json.dumps([str(dest / f'{h:04d}.jpg') for h in history]),
                         'speed':speed,'trajectory':json.dumps(traj.tolist())})
            accepted+=1
        summary.append({'segment':sid,'latitude':latitude,'longitude':longitude,'decoded':decoded,'frame_times':len(ts),'samples':accepted})
        print(f'{sid}: decoded={decoded}, aligned={len(ts)}, examples={accepted}, position=({latitude:.3f},{longitude:.3f})', flush=True)
    csv_tmp = output / 'samples.csv.tmp'
    summary_tmp = output / 'summary.json.tmp'
    with csv_tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['segment','index','history','speed','trajectory'])
        writer.writeheader()
        writer.writerows(rows)
    summary_tmp.write_text(json.dumps(summary, indent=2))
    csv_tmp.replace(csv_path)
    summary_tmp.replace(summary_path)
    print(f'DONE: {len(rows)} training examples across {sum(s["samples"]>0 for s in summary)} segments in {output}')
if __name__=='__main__':main()
