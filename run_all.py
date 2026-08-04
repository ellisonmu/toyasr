"""
run_all.py -- Run model2/train.py for each requested bitdepth.

Usage:
    python run_all.py                    # defaults to bitdepth 3
    python run_all.py --bitdepths 3      # same
    python run_all.py --bitdepths 1 2 3  # runs model2 for each bitdepth in order
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE   = Path(__file__).parent
TRAIN2 = HERE / "model2" / "train.py"


def run(bitdepth: int):
    print(f"\n{'=' * 60}")
    print(f"Running model2/train.py  --bitdepth {bitdepth}")
    print(f"{'=' * 60}\n")
    result = subprocess.run(
        [sys.executable, str(TRAIN2), "--bitdepth", str(bitdepth)],
        cwd=str(HERE),
    )
    if result.returncode != 0:
        print(f"\n[run_all] model2/train.py (bitdepth={bitdepth}) exited with "
              f"code {result.returncode} -- stopping.")
        sys.exit(result.returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bitdepths", type=int, nargs="+", default=[3],
        metavar="N",
        help="One or more bitdepths to sweep (e.g. --bitdepths 1 2 3). Default: 3",
    )
    args = parser.parse_args()

    for bd in args.bitdepths:
        run(bd)

    print("\n[run_all] All done.")
