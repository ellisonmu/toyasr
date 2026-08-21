"""
plot_sweep.py -- summarize a completed evaluation sweep: min WER vs bitdepth,
one line per (dither, mulaw) setting.

Usage (standalone):
    python plot_sweep.py <results_dir>
    python plot_sweep.py results

Also importable: call run(results_dir) from a train script.
"""

import re
import sys
import csv
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt

RUN_DIR_RE = re.compile(r"^(\d+)bit_dither(On|Off)_mulaw(On|Off)$")

def run(results_dir: Path) -> Path:
    """
    Read every results_dir/<N>bit_dither{On,Off}_mulaw{On,Off}/wer_cer_vs_epoch.csv,
    plot min WER vs bitdepth (one line per dither/mulaw setting), save PNG
    alongside the per-run results. Returns the path of the saved PNG.
    """
    results_dir = Path(results_dir)
    series = defaultdict(list)  # (dither, mulaw) -> [(bitdepth, min_wer), ...]

    for run_dir in sorted(results_dir.iterdir()):
        match = RUN_DIR_RE.match(run_dir.name)
        csv_path = run_dir / "wer_cer_vs_epoch.csv"
        if not match or not csv_path.exists():
            continue
        bitdepth, dither, mulaw = int(match.group(1)), match.group(2), match.group(3)

        with open(csv_path, newline="") as f:
            wers = [float(row["wer"]) for row in csv.DictReader(f)]
        if not wers:
            continue
        series[(dither, mulaw)].append((bitdepth, min(wers)))

    control_csv = results_dir / "clean" / "wer_cer_vs_epoch.csv"
    control_min_wer = None
    if control_csv.exists():
        with open(control_csv, newline="") as f:
            control_wers = [float(row["wer"]) for row in csv.DictReader(f)]
        if control_wers:
            control_min_wer = min(control_wers)

    if not series and control_min_wer is None:
        print(f"No run results found under {results_dir} -- skipping sweep plot.")
        return None

    print(f"\n{'dither':>8}  {'compression':>11}  {'bitdepth':>8}  {'min WER':>8}")
    plt.figure()
    for (dither, mulaw), points in sorted(series.items()):
        points.sort()
        bitdepths, min_wers = zip(*points)
        for bd, w in points:
            print(f"{dither:>8}  {mulaw:>11}  {bd:>8}  {w:>8.4f}")
        plt.plot(bitdepths, min_wers, marker="o", label=f"dither={dither}, compression={mulaw}")

    if control_min_wer is not None:
        print(f"{'-':>8}  {'-':>11}  {'control':>8}  {control_min_wer:>8.4f}")
        plt.axhline(control_min_wer, linestyle="--", color="black",
                    label=f"control (no processing): {control_min_wer:.3f}")

    plt.xlabel("Bitdepth")
    plt.ylabel("Min WER (over all epochs)")
    plt.title("Best WER vs bitdepth, by dither/compression setting")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    out_path = results_dir / "min_wer_vs_bitdepth.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Sweep plot saved to {out_path}")
    return out_path

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_sweep.py <results_dir>")
        print("  e.g. python plot_sweep.py results")
        sys.exit(1)
    out = run(Path(sys.argv[1]))
    if out:
        plt.show()
