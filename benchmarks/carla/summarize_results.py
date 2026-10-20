"""Print the README results table from results CSVs.

    python benchmarks/carla/summarize_results.py "expert=results/expert_test_town10.csv" "pilot v1=results/pilot_v1_test_town10.csv"
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.carla.metrics import markdown_table, summarize  # noqa: E402


def main():
    named = {}
    for arg in sys.argv[1:]:
        name, _, path = arg.rpartition("=")
        name = name or Path(path).stem
        with open(path, newline="") as handle:
            named[name] = summarize(list(csv.DictReader(handle)))
    print(markdown_table(named))
    for name, summary in named.items():
        print(f"\n{name}: " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in summary.items()))


if __name__ == "__main__":
    main()
