"""Resumable comma1M download -> preprocess -> train -> cleanup pipeline.

Requires existing patched prepare_comma1m.py, train_comma1m.py,
train_comma1m_v2.py and original V1 split.json.

python pipeline.py --hours 0.167 --batch-segments 2 --epochs 2
python pipeline.py --hours 1 --batch-segments 5 --epochs 3
"""
import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'data/external/comma1m'
PROCESSED = ROOT / 'data/processed/comma1m_v1'
TRACKER = ROOT / 'data/collection_tracker.json'
OUTPUT = ROOT / 'checkpoints/comma1m_pipeline'
V1_SPLIT = ROOT / 'checkpoints/comma1m_cnn_gru_v1/split.json'
V2_BEST = ROOT / 'checkpoints/comma1m_cnn_gru_v2/best.pt'
REPO = 'commaai/comma1M'


def read_json(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2))
    os.replace(temp, path)


def rows_by_segment():
    out = {}
    if (PROCESSED / 'samples.csv').exists():
        with (PROCESSED / 'samples.csv').open(newline='') as file:
            for row in csv.DictReader(file):
                out.setdefault(row['segment'], []).append(row)
    return out


def verify(sid):
    rows = rows_by_segment().get(sid, [])
    summary = read_json(PROCESSED / 'summary.json', [])
    record = next((x for x in summary if x.get('segment') == sid), None)
    if not rows or not record or int(record.get('samples', -1)) != len(rows):
        return 0
    for row in rows:
        try:
            history = json.loads(row['history'])
            trajectory = json.loads(row['trajectory'])
            if len(history) != 4 or len(trajectory) != 20:
                return 0
            if not all(Path(x).is_file() and Path(x).stat().st_size > 0 for x in history):
                return 0
        except (ValueError, TypeError, KeyError, OSError):
            return 0
    return len(rows)


def free_gb():
    return shutil.disk_usage(ROOT).free / 1024**3


def raw_video(sid):
    return RAW / 'data' / sid / 'fcamera.hevc'


def cleanup(sids):
    """Delete only raw videos for segments included in a saved training checkpoint."""
    for sid in sids:
        if verify(sid) and raw_video(sid).is_file():
            raw_video(sid).unlink()
            print('  Removed raw video:', sid, flush=True)


def train(epochs, batch_size, lr):
    import torch
    import numpy as np
    from torch import nn
    from torch.utils.data import DataLoader
    from train_comma1m import DriveSet
    from train_comma1m_v2 import ResidualCNNGRU

    all_rows = rows_by_segment()
    if not all_rows:
        raise RuntimeError('No processed data yet')
    split = read_json(V1_SPLIT, None)
    if not split:
        raise RuntimeError('Missing original V1 split.json')
    val_ids = set(split['validation'])
    old_train = set(split['train'])
    if val_ids & old_train or not val_ids.issubset(all_rows):
        raise RuntimeError('Original validation split missing or overlapping')
    train_ids = set(all_rows) - val_ids
    train_rows = [r for sid in sorted(train_ids) for r in all_rows[sid]]
    val_rows = [r for sid in sorted(val_ids) for r in all_rows[sid]]
    if not train_rows or not val_rows:
        raise RuntimeError('Empty train or validation data')

    torch.manual_seed(27)
    np.random.seed(27)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = ResidualCNNGRU().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    last_file = OUTPUT / 'last.pt'
    prior_epochs = 0
    best = float('inf')
    if last_file.exists():
        ckpt = torch.load(last_file, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        for group in optimizer.param_groups:
            group['lr'] = lr
        prior_epochs = int(ckpt['epoch'])
        best = float(ckpt.get('best_ade', best))
        print('Resuming pipeline checkpoint at epoch', prior_epochs, flush=True)
    elif V2_BEST.exists():
        ckpt = torch.load(V2_BEST, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print('Initialized from original V2 best checkpoint', flush=True)
    else:
        print('Initializing new residual V2 model', flush=True)

    train_loader = DataLoader(DriveSet(train_rows), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(DriveSet(val_rows), batch_size=batch_size, shuffle=False, num_workers=0)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUTPUT / 'split.json', {'train': sorted(train_ids), 'validation': sorted(val_ids)})
    print(f'Training {len(train_rows)} samples; validating {len(val_rows)}; device={device}', flush=True)
    for n in range(1, epochs + 1):
        model.train()
        loss_total = 0.0
        for images, speed, target in train_loader:
            images, speed, target = images.to(device), speed.to(device), target.to(device)
            prediction, _ = model(images, speed)
            loss = nn.functional.smooth_l1_loss(prediction, target, beta=2.0)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_total += loss.item() * len(images)
        model.eval()
        dist_sum = 0.0
        baseline_sum = 0.0
        count = 0
        with torch.inference_mode():
            for images, speed, target in val_loader:
                images, speed, target = images.to(device), speed.to(device), target.to(device)
                prediction, baseline = model(images, speed)
                dist_sum += torch.linalg.vector_norm(prediction - target, dim=-1).sum().item()
                baseline_sum += torch.linalg.vector_norm(baseline - target, dim=-1).sum().item()
                count += len(images)
        ade = dist_sum / (count * 20)
        baseline_ade = baseline_sum / (count * 20)
        improved = ade < best
        best = min(best, ade)
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': prior_epochs + n,
            'val_ade': ade,
            'best_ade': best,
            'trained_segments': sorted(train_ids),
            'validation_segments': sorted(val_ids),
        }
        temp = OUTPUT / 'last.pt.tmp'
        torch.save(checkpoint, temp)
        os.replace(temp, last_file)
        if improved:
            shutil.copy2(last_file, OUTPUT / 'best.pt')
        print(f'Epoch {prior_epochs+n}: loss={loss_total/len(train_rows):.4f} '
              f'val_ADE={ade:.3f}m baseline_ADE={baseline_ade:.3f}m', flush=True)
    return set(train_ids)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hours', type=float, required=False, help='NEW valid one-minute clips to collect this run')
    p.add_argument('--train-only', action='store_true', help='Train on existing data without downloading or cleanup')
    p.add_argument('--batch-segments', type=int, default=5)
    p.add_argument('--epochs', type=int, default=3, help='Training epochs after each batch')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--min-free-gb', type=float, default=5)
    args = p.parse_args()
    if args.batch_segments < 1 or args.epochs < 1 or args.batch_size < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        p.error('Invalid training parameters')
    if args.train_only:
        if args.hours is not None:
            p.error('Do not specify --hours with --train-only')
        print('TRAIN ONLY: No downloading or raw-video cleanup', flush=True)
        train(args.epochs, args.batch_size, args.lr)
        return
    if args.hours is None or not math.isfinite(args.hours) or args.hours <= 0:
        p.error('--hours must be positive unless --train-only is used')
    target = math.ceil(args.hours * 60)
    if not (ROOT / 'prepare_comma1m.py').is_file() or not V1_SPLIT.exists():
        raise SystemExit('Missing existing prepare_comma1m.py or V1 split.json')

    from huggingface_hub import HfApi, hf_hub_download
    import numpy as np
    from pymap3d import ecef2geodetic
    from safetensors.numpy import load_file
    tracker = read_json(TRACKER, {})
    RAW.mkdir(parents=True, exist_ok=True)
    print(f'Existing segments: {len(rows_by_segment())}; collecting +{target}; free {free_gb():.1f} GiB', flush=True)
    print('Fetching segment index...', flush=True)
    segments = sorted(entry.path.split('/')[-1] for entry in HfApi().list_repo_tree(
        REPO, repo_type='dataset', path_in_repo='data', recursive=False)
        if entry.path.startswith('data/'))
    collected = 0
    pending = []
    errors = 0

    def download(sid, filename):
        path = RAW / 'data' / sid / filename
        if path.is_file() and path.stat().st_size:
            return path
        return Path(hf_hub_download(repo_id=REPO, repo_type='dataset',
            filename=f'data/{sid}/{filename}', local_dir=str(RAW)))

    def finish_batch():
        nonlocal pending
        trained = train(args.epochs, args.batch_size, args.lr)
        cleanup(trained)
        for sid in pending:
            tracker[sid] = {**tracker.get(sid, {}), 'status': 'trained', 'samples': verify(sid)}
        atomic_json(TRACKER, tracker)
        pending = []

    try:
        # Recover clips processed before a previous interruption, including user's 2 new clips.
        previous = set()
        if (OUTPUT / 'last.pt').exists():
            import torch
            previous = set(torch.load(OUTPUT / 'last.pt', map_location='cpu', weights_only=False).get('trained_segments', []))
        old_val = set(read_json(V1_SPLIT, {})['validation'])
        untrained = sorted(set(rows_by_segment()) - old_val - previous)
        if untrained:
            print(f'{len(untrained)} existing segments not in pipeline checkpoint: training first', flush=True)
            finish_batch()

        for sid in segments:
            if collected >= target:
                break
            if sid in rows_by_segment():
                continue
            if tracker.get(sid, {}).get('status') in ('outside_us_bounds', 'rejected'):
                continue
            if free_gb() < args.min_free_gb:
                print('STOP: low disk space before download', flush=True)
                break
            print(f'[{collected}/{target}] {sid}', flush=True)
            try:
                loc = download(sid, 'localizer.safetensors')
                arrays = load_file(str(loc))
                states = np.asarray(arrays['frame_states'])
                if states.ndim != 2 or states.shape[1] < 10:
                    raise ValueError('Invalid localization data')
                lat, lon, _ = ecef2geodetic(*states[::20, :3].T)
                latitude, longitude = float(np.median(lat)), float(np.median(lon))
                candidate = np.isfinite([latitude, longitude]).all() and 24 <= latitude <= 49.5 and -125 <= longitude <= -66
                tracker[sid] = {'status': 'us_candidate' if candidate else 'outside_us_bounds',
                                'latitude': latitude, 'longitude': longitude}
                atomic_json(TRACKER, tracker)
                if not candidate:
                    print('  Outside geographic bounds; skip', flush=True)
                    continue
                if free_gb() < args.min_free_gb:
                    print('STOP: low disk space', flush=True)
                    break
                download(sid, 'fcamera.hevc')
                subprocess.run([sys.executable, str(ROOT / 'prepare_comma1m.py'),
                    '--source', str(RAW / 'data'), '--output', str(PROCESSED), '--segment-id', sid],
                    check=True, cwd=ROOT)
                count = verify(sid)
                if not count:
                    tracker[sid] = {**tracker[sid], 'status': 'rejected', 'reason': 'No verified samples'}
                    atomic_json(TRACKER, tracker)
                    # Preserve raw data on preprocessing failure for inspection.
                    print('  No verified samples; skip', flush=True)
                    continue
                tracker[sid] = {**tracker[sid], 'status': 'processed', 'samples': count}
                atomic_json(TRACKER, tracker)
                collected += 1
                pending.append(sid)
                print(f'  +{count} samples; collected {collected}/{target}', flush=True)
                if len(pending) >= args.batch_segments:
                    finish_batch()
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as error:
                errors += 1
                tracker[sid] = {**tracker.get(sid, {}), 'status': 'error', 'reason': str(error)}
                atomic_json(TRACKER, tracker)
                print(f'  ERROR: {error}', flush=True)
        if pending:
            finish_batch()
    except KeyboardInterrupt:
        print('Interrupted; saved progress. Run the same command again to resume.', flush=True)
        return
    print(f'DONE: +{collected}/{target} segments; total={len(rows_by_segment())}; '
          f'errors={errors}; free={free_gb():.1f} GiB; checkpoint={OUTPUT / "last.pt"}', flush=True)


if __name__ == '__main__':
    main()
