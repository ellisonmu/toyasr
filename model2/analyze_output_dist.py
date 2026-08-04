"""
analyze_output_dist.py -- Output-layer distribution analysis for CTCConv2DModel

For each trained condition (alpha sweep at a given bitdepth), loads the
last-epoch weights and runs N_CLIPS validation clips through the model,
collecting the 27-way log-probability output at every non-padded time step.

Three complementary views of model "confidence" are produced:

  char_prob.png          Mean P[token] for blank and each letter (a-z),
                         overlaid across all alpha conditions.
  entropy_kde.png        KDE of per-step Shannon entropy H(t), one curve
                         per alpha.  H=0: certain; H=ln(27)~=3.3: uniform.
  confidence_summary.png Line plots of mean entropy, mean max-prob, and
                         blank-argmax ratio vs alpha.
  output_dist.csv        Per-condition summary statistics.

Aggregation: ~200 validation clips x ~25 time steps each = ~5,000 per-step
observations per condition. Padded frames are excluded via
model.output_length(raw_lengths).

Usage:
    python analyze_output_dist.py --bitdepth 2
"""

import argparse
import csv
import random
import sys
import numpy as np
import torch
from pathlib import Path
from functools import partial
from scipy.stats import gaussian_kde
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_model2   = Path(__file__).parent
_toymodel = _model2.parent
_root     = _toymodel.parent
sys.path.insert(0, str(_toymodel))
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "experiments"))

from model   import CTCConv2DModel
from dataset import SpeechCommandsCTC, ctc_collate, VOCAB_SIZE, BLANK_IDX
from experiments.dithering import dither_app

DATA_ROOT   = _toymodel / "data"
WEIGHTS_DIR = _model2 / "weights"
OUT_DIR     = _model2 / "output_dist"

N_CLIPS     = 200
BATCH_SIZE  = 32
KDE_SAMPLES = 50_000
RNG_SEED    = 42

HIDDEN_DIM = 128
NUM_LAYERS = 2
DEVICE     = torch.device("cpu")

# Token label for each output index: 0=blank, 1=a, ..., 26=z
TOKEN_LABELS = ["<blank>"] + list("abcdefghijklmnopqrstuvwxyz")


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_quantize_fn(bitdepth, alpha):
    if bitdepth is None or alpha is None:
        return None
    return partial(dither_app, bitdepth=bitdepth, param=alpha,
                   Qtype="uniform", Dtype="subtractive")


def discover_conditions(weights_dir: Path, bitdepth: int):
    """
    Scan weights_dir/{bitdepth}bit/*alpha/ for last-epoch checkpoints.
    Returns list of (label, bitdepth, alpha, last_ckpt) sorted by alpha.
    """
    bd_dir = weights_dir / f"{bitdepth}bit"
    if not bd_dir.is_dir():
        raise RuntimeError(f"No weights found for {bitdepth}bit under {weights_dir}")
    conditions = []
    for param_dir in sorted(bd_dir.iterdir()):
        if not param_dir.is_dir():
            continue
        epochs = sorted(
            param_dir.glob("epoch*.pt"),
            key=lambda p: int(p.stem.replace("epoch", "")),
        )
        if not epochs:
            continue
        alpha = float(param_dir.name.replace("alpha", ""))
        label = f"a={alpha}"
        conditions.append((label, bitdepth, alpha, epochs[-1]))
    conditions.sort(key=lambda c: c[2])
    return conditions


def load_model(ckpt_path: Path) -> CTCConv2DModel:
    model = CTCConv2DModel(
        n_mels=80, hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS, vocab_size=VOCAB_SIZE,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    return model


def collect_output_stats(model, loader, n_clips):
    """
    Run up to n_clips clips through model and collect per-time-step statistics
    from the final log-probability output, masking out padded frames.

    Returns:
        entropy_arr    : (N,) -- Shannon entropy per step (nats)
        max_prob_arr   : (N,) -- max softmax probability per step
        argmax_arr     : (N,) int -- argmax token index per step
        mean_char_prob : (VOCAB_SIZE,) -- mean P[token] averaged over all steps
    """
    entropy_list  = []
    max_prob_list = []
    argmax_list   = []
    char_prob_acc = np.zeros(VOCAB_SIZE, dtype=np.float64)
    total_steps   = 0
    seen          = 0

    with torch.no_grad():
        for feats, targets, raw_lengths, target_lengths in loader:
            if seen >= n_clips:
                break
            log_probs     = model(feats.to(DEVICE).transpose(1, 2))  # (B, T', V)
            input_lengths = model.output_length(raw_lengths)          # (B,)

            probs    = log_probs.exp()                                # (B, T', V)
            entropy  = -(probs * log_probs).sum(dim=-1)              # (B, T')
            max_prob = probs.max(dim=-1).values                       # (B, T')
            argmax   = log_probs.argmax(dim=-1)                       # (B, T')

            for b in range(feats.size(0)):
                t = input_lengths[b].item()
                entropy_list.append(entropy[b, :t].cpu().numpy())
                max_prob_list.append(max_prob[b, :t].cpu().numpy())
                argmax_list.append(argmax[b, :t].cpu().numpy())
                char_prob_acc += probs[b, :t, :].cpu().numpy().sum(axis=0)
                total_steps   += t

            seen += feats.size(0)

    entropy_arr    = np.concatenate(entropy_list).astype(np.float32)
    max_prob_arr   = np.concatenate(max_prob_list).astype(np.float32)
    argmax_arr     = np.concatenate(argmax_list).astype(np.int32)
    mean_char_prob = (char_prob_acc / max(total_steps, 1)).astype(np.float32)

    return entropy_arr, max_prob_arr, argmax_arr, mean_char_prob


def compute_stats(entropy_arr, max_prob_arr, argmax_arr):
    return {
        "mean_entropy":       float(np.mean(entropy_arr)),
        "std_entropy":        float(np.std(entropy_arr)),
        "mean_max_prob":      float(np.mean(max_prob_arr)),
        "blank_argmax_ratio": float(np.mean(argmax_arr == BLANK_IDX)),
    }


def kde_curve(values, lo=None, hi_pct=99.0, n_points=512):
    rng = np.random.default_rng(RNG_SEED)
    if len(values) > KDE_SAMPLES:
        values = rng.choice(values, size=KDE_SAMPLES, replace=False)
    lo = float(lo) if lo is not None else float(np.percentile(values, 1.0))
    hi = float(np.percentile(values, hi_pct))
    if hi <= lo:
        hi = lo + 1e-6
    x = np.linspace(lo, hi, n_points)
    try:
        density = gaussian_kde(values, bw_method="scott")(x)
    except np.linalg.LinAlgError:
        density = np.zeros_like(x)
    return x, density


def _alpha_color(alpha, alphas):
    """Map alpha value to a viridis colour, scaled to the observed range."""
    lo, hi = min(alphas), max(alphas)
    t = (alpha - lo) / (hi - lo) if hi > lo else 0.5
    return plt.cm.viridis(0.1 + 0.8 * t)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_char_prob(matched_data, bitdepth, out_dir):
    """
    Two-subplot figure:
      Top:    Mean P[blank] vs alpha -- shows how much output mass is on blank.
      Bottom: Mean P[token] for each letter (a-z), one curve per alpha,
              coloured by alpha value (viridis gradient).
    """
    alphas = [d["alpha"] for d in matched_data]

    fig, (ax_blank, ax_letters) = plt.subplots(
        2, 1, figsize=(12, 7),
        gridspec_kw={"height_ratios": [1, 2]},
    )

    # -- top: blank probability vs alpha --
    blank_probs = [d["mean_char_prob"][BLANK_IDX] for d in matched_data]
    ax_blank.bar(
        [str(d["alpha"]) for d in matched_data], blank_probs,
        color=[_alpha_color(d["alpha"], alphas) for d in matched_data],
        edgecolor="k", linewidth=0.5,
    )
    ax_blank.set_ylabel("Mean P[blank]")
    ax_blank.set_xlabel("Alpha")
    ax_blank.set_title(f"Mean blank-token probability vs alpha  ({bitdepth}-bit)")
    ax_blank.set_ylim(0, 1)
    ax_blank.grid(True, axis="y", alpha=0.3)

    # -- bottom: letter probabilities --
    x = np.arange(1, VOCAB_SIZE)     # indices 1..26
    letter_labels = TOKEN_LABELS[1:]  # a..z
    for d in matched_data:
        color = _alpha_color(d["alpha"], alphas)
        ax_letters.plot(
            x, d["mean_char_prob"][1:],
            marker="o", markersize=3, linewidth=1.2,
            color=color, label=f'a={d["alpha"]}', alpha=0.85,
        )
    ax_letters.set_xticks(x)
    ax_letters.set_xticklabels(letter_labels, fontsize=8)
    ax_letters.set_ylabel("Mean P[token]")
    ax_letters.set_xlabel("Letter token")
    ax_letters.set_title(f"Mean letter-token probability  ({bitdepth}-bit)")
    ax_letters.legend(fontsize=7, ncol=3, loc="upper right")
    ax_letters.grid(True, alpha=0.3)

    sm = plt.cm.ScalarMappable(
        cmap="viridis",
        norm=plt.Normalize(vmin=min(alphas), vmax=max(alphas)),
    )
    sm.set_array([])
    fig.colorbar(sm, ax=ax_letters, label="Alpha", fraction=0.02, pad=0.01)

    fig.tight_layout()
    fig.savefig(out_dir / "char_prob.png", dpi=150)
    plt.close(fig)
    print("  Saved char_prob.png")


def plot_entropy_kde(matched_data, bitdepth, out_dir):
    """
    Overlaid KDE of per-step Shannon entropy H(t) = -sum(p log p).
    Range: 0 (fully certain) to ln(27) ~ 3.3 nats (uniform).
    One curve per alpha, colour-coded by alpha value.
    """
    alphas = [d["alpha"] for d in matched_data]
    max_entropy = float(np.log(VOCAB_SIZE))

    fig, ax = plt.subplots(figsize=(9, 4))
    for d in matched_data:
        color = _alpha_color(d["alpha"], alphas)
        x, dens = kde_curve(d["entropy_arr"], lo=0.0, hi_pct=99.5)
        ax.plot(x, dens, color=color, linewidth=1.5,
                label=f'a={d["alpha"]}', alpha=0.85)

    ax.axvline(max_entropy, color="k", linestyle=":", linewidth=1,
               label=f"max entropy  ln(27)={max_entropy:.2f}")
    ax.set_xlabel("Entropy (nats)  [0=certain, ln(27)=uniform]")
    ax.set_ylabel("Density")
    ax.set_title(f"Per-step output entropy KDE  ({bitdepth}-bit, matched eval)\n"
                 "Curve shifted left = model is more confident at that alpha")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)

    sm = plt.cm.ScalarMappable(
        cmap="viridis",
        norm=plt.Normalize(vmin=min(alphas), vmax=max(alphas)),
    )
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Alpha", fraction=0.02, pad=0.01)

    fig.tight_layout()
    fig.savefig(out_dir / "entropy_kde.png", dpi=150)
    plt.close(fig)
    print("  Saved entropy_kde.png")


def plot_confidence_summary(matched_data, bitdepth, out_dir):
    """
    Three-subplot line figure, all vs alpha:
      1. Mean entropy       -- lower = more confident overall
      2. Mean max-prob      -- higher = more peaked distributions
      3. Blank-argmax ratio -- fraction of steps where blank is the top token
    """
    alphas        = [d["alpha"] for d in matched_data]
    mean_entropy  = [d["stats"]["mean_entropy"]       for d in matched_data]
    mean_max_prob = [d["stats"]["mean_max_prob"]      for d in matched_data]
    blank_ratio   = [d["stats"]["blank_argmax_ratio"] for d in matched_data]

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    for ax, values, ylabel, title in [
        (axes[0], mean_entropy,  "Mean entropy (nats)",
         "Mean per-step entropy  [lower = more confident]"),
        (axes[1], mean_max_prob, "Mean max probability",
         "Mean max softmax probability  [higher = more peaked]"),
        (axes[2], blank_ratio,   "Blank-argmax ratio",
         "Fraction of steps where blank is top-1 token"),
    ]:
        ax.plot(alphas, values, "o-", color="#1f77b4",
                linewidth=2, markersize=6)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.3)
        for a, v in zip(alphas, values):
            ax.annotate(f"{v:.3f}", (a, v),
                        textcoords="offset points", xytext=(4, 4), fontsize=7)

    axes[2].set_xlabel("Dither alpha")
    fig.suptitle(f"Output confidence summary  ({bitdepth}-bit, matched eval)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "confidence_summary.png", dpi=150)
    plt.close(fig)
    print("  Saved confidence_summary.png")


def save_csv(matched_data, out_dir):
    path = out_dir / "output_dist.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "label", "bitdepth", "alpha",
            "mean_entropy", "std_entropy", "mean_max_prob", "blank_argmax_ratio",
        ])
        for d in matched_data:
            s = d["stats"]
            writer.writerow([
                d["label"], d["bitdepth"], d["alpha"],
                f"{s['mean_entropy']:.6f}",
                f"{s['std_entropy']:.6f}",
                f"{s['mean_max_prob']:.6f}",
                f"{s['blank_argmax_ratio']:.6f}",
            ])
    print("  Saved output_dist.csv")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bitdepth", type=int, default=2)
    args = parser.parse_args()
    bitdepth = args.bitdepth

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    conditions = discover_conditions(WEIGHTS_DIR, bitdepth)
    if not conditions:
        raise RuntimeError(f"No conditions found for {bitdepth}-bit")
    print(f"Found {len(conditions)} conditions for {bitdepth}-bit:")
    for label, bd, alpha, ckpt in conditions:
        print(f"  {label:12s}  last epoch: {ckpt.stem}")

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

    print(f"\n-- Collecting output statistics (MATCHED) -----------------------")
    matched_data = []
    for label, bd, alpha, ckpt in conditions:
        print(f"  {label}  [{ckpt.stem}]")
        model = load_model(ckpt)
        val_ds.quantize_fn = make_quantize_fn(bd, alpha)

        entropy_arr, max_prob_arr, argmax_arr, mean_char_prob = \
            collect_output_stats(model, make_loader(), N_CLIPS)

        stats = compute_stats(entropy_arr, max_prob_arr, argmax_arr)
        print(f"    steps={len(entropy_arr):,}  "
              f"mean_H={stats['mean_entropy']:.3f}  "
              f"mean_maxP={stats['mean_max_prob']:.3f}  "
              f"blank_ratio={stats['blank_argmax_ratio']:.3f}")

        matched_data.append({
            "label": label, "bitdepth": bd, "alpha": alpha,
            "entropy_arr":    entropy_arr,
            "max_prob_arr":   max_prob_arr,
            "mean_char_prob": mean_char_prob,
            "stats":          stats,
        })

    print(f"\n-- Generating plots ----------------------------------------------")
    plot_char_prob(matched_data, bitdepth, OUT_DIR)
    plot_entropy_kde(matched_data, bitdepth, OUT_DIR)
    plot_confidence_summary(matched_data, bitdepth, OUT_DIR)
    save_csv(matched_data, OUT_DIR)

    print(f"\n{'Label':<15} {'Entropy':>10} {'MaxProb':>10} {'BlankRatio':>12}")
    print("-" * 50)
    for d in matched_data:
        s = d["stats"]
        print(f"{d['label']:<15} {s['mean_entropy']:>10.4f} "
              f"{s['mean_max_prob']:>10.4f} {s['blank_argmax_ratio']:>12.4f}")

    print(f"\nAll outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
