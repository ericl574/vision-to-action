"""Download a bounded number of comma1M one-minute driving segments.

Install: python -m pip install -U huggingface_hub
Usage: python download_real_driving.py --segments 10 --budget-gb 3
This DOES NOT filter for North America; use localization GPS for that afterward.
"""
import argparse
import json
import shutil
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

REPO = "commaai/comma1M"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--segments", type=int, default=10)
    p.add_argument("--budget-gb", type=float, default=3.0)
    p.add_argument("--out", type=Path, default=Path("data/external/comma1m"))
    a = p.parse_args()
    if a.segments < 1 or a.budget_gb <= 0:
        p.error("segments and budget-gb must be positive")
    a.out.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    segments = sorted(x.path.removeprefix("data/") for x in api.list_repo_tree(REPO, repo_type="dataset", path_in_repo="data", recursive=False) if x.path.startswith("data/"))
    if not segments:
        raise RuntimeError("No segment directories were returned by Hugging Face")
    manifest_path = a.out / "download_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    budget = int(a.budget_gb * (1024**3))
    def size_on_disk():
        return sum(f.stat().st_size for f in a.out.rglob('*') if f.is_file())
    completed = sum(v.get("status") == "complete" for v in manifest.values())
    for seg in segments:
        if completed >= a.segments:
            break
        if manifest.get(seg, {}).get("status") == "complete":
            continue
        if shutil.disk_usage(a.out).free < 3 * 1024**3:
            print("STOP: less than 3 GiB free disk space")
            break
        if size_on_disk() >= budget:
            print("STOP: download budget reached")
            break
        print(f"Downloading segment {seg} ({completed+1}/{a.segments})", flush=True)
        paths = []
        try:
            for filename in ["thumbnail.jpg", "localizer.safetensors", "fcamera.hevc"]:
                rel = f"data/{seg}/{filename}"
                dest = hf_hub_download(repo_id=REPO, repo_type="dataset", filename=rel, local_dir=str(a.out))
                paths.append(str(Path(dest).relative_to(a.out)))
            manifest[seg] = {"status": "complete", "files": paths}
            completed += 1
        except Exception as ex:
            manifest[seg] = {"status": "incomplete", "error": str(ex)}
            print(f"Failed {seg}: {ex}")
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"Complete segments: {completed}; local files: {size_on_disk() / 1024**3:.2f} GiB", flush=True)
    print(f"DONE: {completed} complete segments; directory: {a.out.resolve()}")
    print("NOTE: Geographic filtering, timestamp alignment, and trajectory label validation are still required before training.")

if __name__ == "__main__":
    main()
