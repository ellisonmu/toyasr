"""
plot_sweep.py -- plot min WER / min CER vs dither alpha across a completed sweep.

Usage (standalone):
    python plot_sweep.py <results_dir> <bitdepth>
    python plot_sweep.py model2/results 3
    python plot_sweep.py model1/results 2

Also importable: call run(results_dir, bitdepth) from a train script.
"""

import sys
import csv
from pathlib import Path
import matplotlib.pyplot as plt


def run(results_dir: Path, bitdepth: int) -> Path:
    """
    Read all *alpha/wer_cer_vs_epoch.csv files under results_dir/{bitdepth}bit/,
    plot min WER and min CER vs alpha, save PNG alongside the CSVs.
    Returns the path of the saved PNG.
    """
    bit_dir = Path(results_dir) / f"{bitdepth}bit"
    csv_paths = sorted(bit_dir.glob("*alpha/wer_cer_vs_epoch.csv"))
    if not csv_paths:
        print(f"No CSVs found under {bit_dir} — skipping sweep plot.")
        return None

    alphas, min_wers, min_cers = [], [], []
    for csv_path in csv_paths:
        alpha_str = csv_path.parent.name.removesuffix("alpha")
        alpha = float(alpha_str)
        wers, cers = [], []
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                wers.append(float(row["wer"]))
                cers.append(float(row["cer"]))
        alphas.append(alpha)
        min_wers.append(min(wers))
        min_cers.append(min(cers))

    pairs = sorted(zip(alphas, min_wers, min_cers))
    alphas, min_wers, min_cers = zip(*pairs)

    print(f"\n{'alpha':>6}  {'min WER':>8}  {'min CER':>8}")
    for a, w, c in zip(alphas, min_wers, min_cers):
        print(f"{a:>6.1f}  {w:>8.4f}  {c:>8.4f}")

    out_path = bit_dir / "min_wer_cer_vs_alpha.png"
    plt.figure()
    plt.plot(alphas, min_wers, marker="o", label="min WER")
    plt.plot(alphas, min_cers, marker="o", label="min CER")
    plt.xlabel("Dither alpha")
    plt.ylabel("Min error rate (over all epochs)")
    plt.title(f"Best WER/CER vs dither alpha  (bitdepth={bitdepth})")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path)
    plt.close()
    print(f"Sweep plot saved to {out_path}")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python plot_sweep.py <results_dir> <bitdepth>")
        print("  e.g. python plot_sweep.py model2/results 3")
        sys.exit(1)
    _results_dir = Path(sys.argv[1])
    _bitdepth    = int(sys.argv[2])
    out = run(_results_dir, _bitdepth)
    if out:
        plt.show()
