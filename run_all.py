"""
run_all.py -- Run train.py across the evaluation sweep: bitdepth x dither
on/off x mu-law compression on/off (6 x 2 x 2 = 24 models by default).

Usage:
    python run_all.py                                # full 24-point sweep
    python run_all.py --bitdepths 3 5                 # restrict bitdepths
    python run_all.py --dither on --mulaw off          # restrict to one dither/mulaw combo
    python run_all.py --bitdepths 3 --dither on off --mulaw on off
"""

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

HERE  = Path(__file__).parent
TRAIN = HERE / "train.py"

ON_OFF = {"on": True, "off": False}

def run(bitdepth: int, dither: bool, mulaw: bool):
    run_label = f"{bitdepth}bit_dither{'On' if dither else 'Off'}_mulaw{'On' if mulaw else 'Off'}"
    print(f"\n{'=' * 60}")
    print(f"Running train.py  --bitdepth {bitdepth} "
          f"--{'dither' if dither else 'no-dither'} --{'mulaw' if mulaw else 'no-mulaw'}"
          f"   ({run_label})")
    print(f"{'=' * 60}\n")
    cmd = [
        sys.executable, str(TRAIN),
        "--bitdepth", str(bitdepth),
        "--dither" if dither else "--no-dither",
        "--mulaw" if mulaw else "--no-mulaw",
    ]
    result = subprocess.run(cmd, cwd=str(HERE))
    if result.returncode != 0:
        print(f"\n[run_all] train.py ({run_label}) exited with "
              f"code {result.returncode} -- stopping.")
        sys.exit(result.returncode)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bitdepths", type=int, nargs="+", default=list(range(1, 7)),
        metavar="N", choices=range(1, 7),
        help="Bitdepths to sweep (1-6). Default: 1 2 3 4 5 6",
    )
    parser.add_argument(
        "--dither", nargs="+", choices=["on", "off"], default=["on", "off"],
        help="Dither settings to sweep. Default: on off",
    )
    parser.add_argument(
        "--mulaw", nargs="+", choices=["on", "off"], default=["on", "off"],
        help="Mu-law companding settings to sweep. Default: on off",
    )
    args = parser.parse_args()

    dither_settings = [ON_OFF[d] for d in args.dither]
    mulaw_settings  = [ON_OFF[m] for m in args.mulaw]

    combos = list(itertools.product(args.bitdepths, dither_settings, mulaw_settings))
    print(f"[run_all] Sweeping {len(combos)} run(s).")

    for bd, dither, mulaw in combos:
        run(bd, dither, mulaw)

    print("\n[run_all] All done.")

    try:
        import plot_sweep
        plot_sweep.run(HERE / "results")
    except Exception as e:
        print(f"[run_all] Skipping summary plot: {e}")
