"""Bounded download of comma1M video and localization pairs from Hugging Face.

Example: python download_real_driving_fixed.py --segments 10 --budget-gb 3
Does not verify geographic provenance or create training labels.
"""
import argparse
import json
import shutil
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

REPO = 'commaai/comma1M'
NEEDED = ('fcamera.hevc', 'localizer.safetensors')
GIB = 1024 ** 3


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--segments', type=int, default=10)
    p.add_argument('--budget-gb', type=float, default=3.0)
    p.add_argument('--min-free-gb', type=float, default=4.0)
    p.add_argument('--out', type=Path, default=Path('data/external/comma1m'))
    args = p.parse_args()
    if args.segments < 1 or args.budget_gb <= 0:
        p.error('segments and budget-gb must be positive')
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest_file = out / 'download_manifest.json'
    manifest = json.loads(manifest_file.read_text()) if manifest_file.exists() else {}
    api = HfApi()
    folders = sorted(entry.path for entry in api.list_repo_tree(REPO, repo_type='dataset', path_in_repo='data', recursive=False) if entry.path.startswith('data/'))
    if not folders:
        raise RuntimeError('No dataset segment directories found')
    done = sum(x.get('status') == 'complete' for x in manifest.values())
    max_bytes = int(args.budget_gb * GIB)
    def local_size():
        return sum(f.stat().st_size for f in out.rglob('*') if f.is_file())
    for folder in folders:
        segment = folder.removeprefix('data/')
        if done >= args.segments:
            break
        if manifest.get(segment, {}).get('status') == 'complete':
            continue
        present = local_size()
        free = shutil.disk_usage(out).free
        if present >= max_bytes or free < int(args.min_free_gb * GIB):
            print(f'STOP: local files {present/GIB:.2f} GiB, free {free/GIB:.2f} GiB')
            break
        print(f'Downloading {segment} ({done+1}/{args.segments}); free {free/GIB:.2f} GiB', flush=True)
        files = []
        try:
            for filename in NEEDED:
                relative = f'{folder}/{filename}'
                # Some dataset segments can omit a file; fail fast and report it.
                dest = Path(hf_hub_download(repo_id=REPO, repo_type='dataset', filename=relative, local_dir=str(out)))
                if not dest.is_file() or dest.stat().st_size == 0:
                    raise RuntimeError(f'Downloaded file is empty: {relative}')
                files.append(str(dest.relative_to(out)))
            manifest[segment] = {'status': 'complete', 'files': files}
            done += 1
        except KeyboardInterrupt:
            print('\nInterrupted. Existing files were retained for resuming.')
            raise
        except Exception as exc:
            manifest[segment] = {'status': 'incomplete', 'error': str(exc)}
            print(f'STOP: download failed for {segment}: {exc}', flush=True)
            break  # never loop through hundreds of failing segments
        finally:
            manifest_file.write_text(json.dumps(manifest, indent=2))
        print(f'Completed {done}; local data {local_size()/GIB:.2f} GiB', flush=True)
    print(f'DONE: {done} complete segments; data in {out}')
    print('Geographic filtering and validation of trajectory timestamps/coordinates still required.')

if __name__ == '__main__':
    main()
