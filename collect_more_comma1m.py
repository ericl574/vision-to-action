
"""
Incremental comma1M dataset collector.

Examples:
    python collect_more_comma1m.py --hours 0.167
    python collect_more_comma1m.py --hours 1
    python collect_more_comma1m.py --hours 5

Each --hours value means ADDITIONAL successful one-minute segments.
This collects and preprocesses data; it does not train the model.
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

import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from pymap3d import ecef2geodetic
from safetensors.numpy import load_file


REPO = "commaai/comma1M"
ROOT = Path(__file__).resolve().parent

RAW = ROOT / "data/external/comma1m"
PROCESSED = ROOT / "data/processed/comma1m_v1"
SAMPLES = PROCESSED / "samples.csv"
SUMMARY = PROCESSED / "summary.json"
TRACKER = ROOT / "data/collection_tracker.json"

MIN_FREE_GB = 5.0


def save_tracker(tracker):
    TRACKER.parent.mkdir(parents=True, exist_ok=True)
    temp = TRACKER.with_suffix(".json.tmp")
    temp.write_text(json.dumps(tracker, indent=2))
    os.replace(temp, TRACKER)


def load_tracker():
    if TRACKER.exists():
        return json.loads(TRACKER.read_text())
    return {}


def load_samples():
    if not SAMPLES.exists():
        return {}

    result = {}

    with SAMPLES.open(newline="") as f:
        for row in csv.DictReader(f):
            sid = row["segment"]
            result.setdefault(sid, []).append(row)

    return result


def verify_samples(sid):
    samples = load_samples().get(sid, [])

    if not samples:
        return 0

    if not SUMMARY.exists():
        return 0

    summaries = {
        x["segment"]: x
        for x in json.loads(SUMMARY.read_text())
    }

    if sid not in summaries:
        return 0

    for row in samples:
        try:
            history = json.loads(row["history"])
            trajectory = json.loads(row["trajectory"])

            if len(history) != 4 or len(trajectory) != 20:
                return 0

            for filename in history:
                path = Path(filename)
                if not path.is_absolute():
                    path = ROOT / path

                if not path.is_file() or path.stat().st_size == 0:
                    return 0

        except (KeyError, ValueError, TypeError, OSError):
            return 0

    return len(samples)


def free_space_gb():
    return shutil.disk_usage(ROOT).free / (1024 ** 3)


def download_file(sid, filename):
    path = RAW / "data" / sid / filename

    if path.exists() and path.stat().st_size > 0:
        return path

    return Path(
        hf_hub_download(
            repo_id=REPO,
            repo_type="dataset",
            filename=f"data/{sid}/{filename}",
            local_dir=str(RAW),
        )
    )


def check_location(sid):
    path = download_file(sid, "localizer.safetensors")

    data = load_file(str(path))
    states = np.asarray(data["frame_states"])

    if states.ndim != 2 or states.shape[1] < 3:
        raise ValueError("Invalid GPS state shape")

    lat, lon, _ = ecef2geodetic(*states[:, :3].T)

    latitude = float(np.median(lat))
    longitude = float(np.median(lon))

    if not np.isfinite([latitude, longitude]).all():
        raise ValueError("Invalid GPS coordinates")

    # Coarse continental-US bounding box.
    # This does not establish national borders precisely.
    candidate = (
        24 <= latitude <= 49.5
        and -125 <= longitude <= -66
    )

    return candidate, latitude, longitude


def preprocess(sid):
    command = [
        sys.executable,
        str(ROOT / "prepare_comma1m.py"),
        "--source",
        str(RAW / "data"),
        "--output",
        str(PROCESSED),
        "--segment-id",
        sid,
    ]

    subprocess.run(command, cwd=ROOT, check=True)


def remove_raw_video(sid):
    video = RAW / "data" / sid / "fcamera.hevc"

    if video.exists():
        size_mb = video.stat().st_size / (1024 ** 2)
        video.unlink()
        print(f"  Cleaned raw video: {size_mb:.1f} MB")

    # Keep the small GPS file for debugging and provenance.


def list_segments():
    api = HfApi()

    folders = api.list_repo_tree(
        REPO,
        repo_type="dataset",
        path_in_repo="data",
        recursive=False,
    )

    return sorted(
        entry.path.split("/")[-1]
        for entry in folders
        if entry.path.startswith("data/")
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hours",
        type=float,
        required=True,
        help="Additional hours of valid one-minute segments to collect",
    )

    args = parser.parse_args()

    if not math.isfinite(args.hours) or args.hours <= 0:
        parser.error("--hours must be positive and finite")

    target = math.ceil(args.hours * 60)

    if not (ROOT / "prepare_comma1m.py").exists():
        raise RuntimeError("prepare_comma1m.py not found")

    tracker = load_tracker()

    # Existing processed samples are always the source of truth.
    existing = load_samples()
    completed = set(existing)

    for sid, rows in existing.items():
        tracker.setdefault(sid, {})
        tracker[sid]["status"] = "verified"
        tracker[sid]["samples"] = len(rows)

    save_tracker(tracker)

    print("=" * 55)
    print("comma1M AUTOMATIC DATA COLLECTOR")
    print("=" * 55)
    print(f"Existing processed segments: {len(completed)}")
    print(f"Existing training samples: {sum(map(len, existing.values()))}")
    print(f"Additional target: {target} segments")
    print(f"Estimated additional duration: {target / 60:.2f} hours")
    print(f"Available disk space: {free_space_gb():.2f} GB")
    print("=" * 55, flush=True)

    segments = list_segments()

    print(f"Dataset segment folders found: {len(segments)}")

    added = 0
    skipped = 0
    errors = 0

    for sid in segments:
        if added >= target:
            break

        if sid in completed:
            continue

        info = tracker.get(sid, {})
        status = info.get("status")

        if status in ("outside_us_bounds", "rejected"):
            skipped += 1
            continue

        print(
            f"\n[{added}/{target} collected] "
            f"Checking {sid}",
            flush=True,
        )

        try:
            # Recover a previously processed segment after interruption.
            if status == "verified":
                count = verify_samples(sid)

                if count:
                    completed.add(sid)
                    remove_raw_video(sid)
                    continue

            if status == "us_candidate":
                candidate = True
                latitude = info.get("latitude")
                longitude = info.get("longitude")
            else:
                candidate, latitude, longitude = check_location(sid)

                tracker[sid] = {
                    "status": (
                        "us_candidate"
                        if candidate
                        else "outside_us_bounds"
                    ),
                    "latitude": latitude,
                    "longitude": longitude,
                }
                save_tracker(tracker)

            if not candidate:
                print("  Skipped: outside US bounds")
                skipped += 1
                continue

            print(
                f"  US candidate: {latitude:.3f}, "
                f"{longitude:.3f}"
            )

            if free_space_gb() < MIN_FREE_GB:
                print("\nSTOP: Disk space is too low.")
                break

            print("  Downloading video...", flush=True)
            download_file(sid, "fcamera.hevc")

            print("  Preprocessing...", flush=True)
            preprocess(sid)

            print("  Verifying samples...", flush=True)
            count = verify_samples(sid)

            if count == 0:
                tracker[sid] = {
                    **tracker.get(sid, {}),
                    "status": "rejected",
                    "reason": "No valid samples produced",
                }
                save_tracker(tracker)

                print("  Rejected: no valid samples")
                skipped += 1
                # Keep raw video if preprocessing produced no samples.
                continue

            # Durable tracking before deleting raw data.
            tracker[sid] = {
                **tracker.get(sid, {}),
                "status": "verified",
                "samples": count,
            }
            save_tracker(tracker)

            completed.add(sid)
            added += 1

            remove_raw_video(sid)

            print(
                f"  SUCCESS: +{count} training samples "
                f"({added}/{target} new segments)",
                flush=True,
            )

        except KeyboardInterrupt:
            print("\nInterrupted. Progress already saved.")
            save_tracker(tracker)
            return

        except Exception as exc:
            errors += 1

            tracker[sid] = {
                **tracker.get(sid, {}),
                "status": "error",
                "reason": str(exc),
            }
            save_tracker(tracker)

            print(f"  ERROR: {exc}", flush=True)
            print("  Skipping to next segment...", flush=True)

    final = load_samples()

    print("\n" + "=" * 55)
    print("BATCH COMPLETE")
    print("=" * 55)
    print(f"New segments collected: {added}/{target}")
    print(f"Total processed segments: {len(final)}")
    print(f"Total training samples: {sum(map(len, final.values()))}")
    print(f"Skipped or rejected this run: {skipped}")
    print(f"Errors this run: {errors}")
    print(f"Free disk space: {free_space_gb():.2f} GB")
    print(f"Tracker: {TRACKER}")
    print("=" * 55)


if __name__ == "__main__":
    main()
