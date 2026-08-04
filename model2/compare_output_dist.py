"""
compare_output_dist.py -- Standard vs mu-law dithering output-distribution comparison

Runs the same analysis as analyze_output_dist.py for both weight variants
(standard and mu-law) at a given bitdepth, then produces a head-to-head
comparison plot overlaying the two across all alpha values.

Outputs written to model2/output_dist_compare/:
  comparison_summary.png   3-metric line plot, standard vs mu-law
  comparison.csv           Combined per-condition stats for both variants

Usage:
    python compare_output_dist.py --bitdepth 3

Note: Only 3-bit has full data for both variants (11 alphas each).
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

_model2   = Path(__file__).parent
_toymodel = _model2.parent
_root     = _toymodel.parent
sys.path.insert(0, str(_model2))
sys.path.insert(0, str(_toymodel))
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "experiments"))

# Import helpers from the sibling script — no code runs at import time
# (main() is guarded by __name__ == "__main__")
from analyze_output_dist import (
    make_quantize_fn,
    discover_conditions,
    load_model,
    collect_output_stats,
    compute_stats,
    DATA_ROOT,
    WEIGHTS_DIR,
    N_CLIPS,
    BATCH_SIZE,
    RNG_SEED,
)
from dataset import SpeechCommandsCTC, ctc_collate

OUT_DIR = _model2 / "output_dist_compare"


# ── Per-variant collection ────────────────────────────────────────────────────

def run_variant(variant_name: str, weights_subdir: str, bitdepth: int,
                val_ds, make_loader_fn):
    """
    Discover conditions under WEIGHTS_DIR/weights_subdir/{bitdepth}bit/,
    run inference for each, and return a list of result dicts.
    """
    conditions = discover_conditions(WEIGHTS_DIR / weights_subdir, bitdepth)
    if not conditions:
        raise RuntimeError(
            f"No conditions found for {variant_name} {bitdepth}-bit "
            f"under {WEIGHTS_DIR / weights_subdir}"
        )
    print(f"\n{variant_name}: {len(conditions)} conditions")
    for label, _, alpha, ckpt in conditions:
        print(f"  {label:12s}  {ckpt.stem}")

    results = []
    print(f"\n-- Collecting stats for {variant_name} --")
    for label, bd, alpha, ckpt in conditions:
        print(f"  {label}  [{ckpt.stem}]")
        model = load_model(ckpt)
        val_ds.quantize_fn = make_quantize_fn(bd, alpha)

        entropy_arr, max_prob_arr, argmax_arr, mean_char_prob = \
            collect_output_stats(model, make_loader_fn(), N_CLIPS)

        stats = compute_stats(entropy_arr, max_prob_arr, argmax_arr)
        print(f"    steps={len(entropy_arr):,}  "
              f"mean_H={stats['mean_entropy']:.3f}  "
              f"mean_maxP={stats['mean_max_prob']:.3f}  "
              f"blank_ratio={stats['blank_argmax_ratio']:.3f}")

        results.append({
            "label":          label,
            "bitdepth":       bd,
            "alpha":          alpha,
            "entropy_arr":    entropy_arr,
            "max_prob_arr":   max_prob_arr,
            "mean_char_prob": mean_char_prob,
            "stats":          stats,
        })

    return results


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_comparison(std_data, mulaw_data, bitdepth, out_dir):
    """
    Three-subplot figure (mean entropy / mean max-prob / blank-argmax ratio)
    vs alpha, with standard (solid blue) and mu-law (dashed orange) overlaid.
    """
    std_alphas   = [d["alpha"] for d in std_data]
    mulaw_alphas = [d["alpha"] for d in mulaw_data]

    metrics = [
        ("mean_entropy",       "Mean entropy (nats)",
         "Mean per-step entropy  [lower = more confident]"),
        ("mean_max_prob",      "Mean max probability",
         "Mean max softmax probability  [higher = more peaked]"),
        ("blank_argmax_ratio", "Blank-argmax ratio",
         "Fraction of steps where blank is top-1 token"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    for ax, (key, ylabel, title) in zip(axes, metrics):
        std_vals   = [d["stats"][key] for d in std_data]
        mulaw_vals = [d["stats"][key] for d in mulaw_data]

        ax.plot(std_alphas, std_vals,
                "o-", color="#1f77b4", linewidth=2, markersize=6,
                label="Standard")
        ax.plot(mulaw_alphas, mulaw_vals,
                "s--", color="#ff7f0e", linewidth=2, markersize=6,
                label="Mu-law")

        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        for a, v in zip(std_alphas, std_vals):
            ax.annotate(f"{v:.3f}", (a, v),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=6, color="#1f77b4")
        for a, v in zip(mulaw_alphas, mulaw_vals):
            ax.annotate(f"{v:.3f}", (a, v),
                        textcoords="offset points", xytext=(4, -10),
                        fontsize=6, color="#ff7f0e")

    axes[2].set_xlabel("Dither alpha")
    fig.suptitle(
        f"Output confidence comparison  ({bitdepth}-bit)\n"
        "Standard dithering vs Mu-law dithering",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_summary.png", dpi=150)
    plt.close(fig)
    print("  Saved comparison_summary.png")


# ── CSV ───────────────────────────────────────────────────────────────────────

def save_combined_csv(std_data, mulaw_data, out_dir):
    path = out_dir / "comparison.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "variant", "label", "bitdepth", "alpha",
            "mean_entropy", "std_entropy", "mean_max_prob", "blank_argmax_ratio",
        ])
        for variant, data in [("standard", std_data), ("mu-law", mulaw_data)]:
            for d in data:
                s = d["stats"]
                writer.writerow([
                    variant, d["label"], d["bitdepth"], d["alpha"],
                    f"{s['mean_entropy']:.6f}",
                    f"{s['std_entropy']:.6f}",
                    f"{s['mean_max_prob']:.6f}",
                    f"{s['blank_argmax_ratio']:.6f}",
                ])
    print("  Saved comparison.csv")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bitdepth", type=int, default=3,
        help="Bitdepth to compare (default 3; only 3-bit has full data for both variants)",
    )
    args = parser.parse_args()
    bitdepth = args.bitdepth

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== Standard vs Mu-law comparison  ({bitdepth}-bit) ===")

    val_ds = SpeechCommandsCTC(DATA_ROOT, subset="validation", quantize_fn=None)
    rng    = random.Random(RNG_SEED)
    idxs   = list(range(len(val_ds)))
    rng.shuffle(idxs)
    val_subset = Subset(val_ds, idxs[:N_CLIPS])

    def make_loader():
        return DataLoader(
            val_subset, batch_size=BATCH_SIZE, shuffle=False,
            collate_fn=ctc_collate, num_workers=0,
        )

    std_data   = run_variant("Standard", "standard", bitdepth, val_ds, make_loader)
    mulaw_data = run_variant("Mu-law",   "mu-law",   bitdepth, val_ds, make_loader)

    print(f"\n-- Generating plots --")
    plot_comparison(std_data, mulaw_data, bitdepth, OUT_DIR)
    save_combined_csv(std_data, mulaw_data, OUT_DIR)

    print(f"\n{'Variant':<12} {'Label':<12} {'Entropy':>10} "
          f"{'MaxProb':>10} {'BlankRatio':>12}")
    print("-" * 60)
    for variant, data in [("standard", std_data), ("mu-law", mulaw_data)]:
        for d in data:
            s = d["stats"]
            print(f"{variant:<12} {d['label']:<12} {s['mean_entropy']:>10.4f} "
                  f"{s['mean_max_prob']:>10.4f} {s['blank_argmax_ratio']:>12.4f}")

    print(f"\nAll outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
