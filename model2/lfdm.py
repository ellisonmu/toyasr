"""
lfdm.py -- Latent Feature Distribution Mapping for CTCConv2DModel

For each trained condition (bitdepth × alpha), loads the final-epoch weights
and runs N_CLIPS validation clips through the model, capturing activations at
two hook points:

  h_conv  after model.conv   (B, 32, 40, T') -- 2D CNN front end
  h_lstm  after model.lstm   (B, T', 256)    -- BiLSTM encoder

Computes variance and excess kurtosis (Normal dist. = 0) of all activation
values flattened across B, channels, and frames, and plots the empirical PDF
via kernel density estimation.

Two modes run in sequence:
  MATCHED  -- each model is evaluated on its own training condition.
              Shows how different quantization regimes shape the learned space.
  CROSS    -- one reference model is evaluated on every input condition,
              including clean (no quantization).
              Directly answers: does the CNN filter out dither noise, or does
              the variance/kurtosis track the input quality all the way through?

Outputs (saved to model2/lfdm/):
  pdf_conv.png         KDE overlays at CNN output, cross-condition
  pdf_lstm.png         KDE overlays at LSTM output, cross-condition
  stats_vs_alpha.png   variance and excess kurtosis vs. alpha (matched)
  rel_var.png          variance normalised to clean input (cross, both layers)
  summary.csv          per-condition statistics table
"""

import csv
import random
import sys
import numpy as np
import torch
from pathlib import Path
from functools import partial
from scipy.stats import gaussian_kde
from scipy.stats import kurtosis as scipy_kurtosis
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── sys.path ──────────────────────────────────────────────────────────────────
_model2   = Path(__file__).parent          # toymodel/model2/
_toymodel = _model2.parent                 # toymodel/
_root     = _toymodel.parent               # python-files/
sys.path.insert(0, str(_toymodel))
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "experiments"))

from model   import CTCConv2DModel
from dataset import SpeechCommandsCTC, ctc_collate, VOCAB_SIZE
from experiments.dithering import dither_app

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_ROOT   = _toymodel / "data"
WEIGHTS_DIR = _model2 / "weights"
OUT_DIR     = _model2 / "lfdm"

N_CLIPS     = 200    # validation clips per condition (shuffled, seed=42)
BATCH_SIZE  = 32
KDE_SAMPLES = 100_000   # subsample cap for gaussian_kde (speed)
RNG_SEED    = 42

# The cross-condition reference: the trained model used when varying input
# conditions.  Point to the condition you expect performed best.
REFERENCE_CONDITION = ("3bit", "0.2alpha")

HIDDEN_DIM = 128
NUM_LAYERS = 2
DEVICE     = torch.device("cpu")
# ──────────────────────────────────────────────────────────────────────────────


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_quantize_fn(bitdepth, alpha):
    if bitdepth is None or alpha is None:
        return None
    return partial(dither_app, bitdepth=bitdepth, param=alpha,
                   Qtype="uniform", Dtype="subtractive")


def discover_conditions(weights_dir: Path):
    """
    Scan weights_dir/{bitdepth}bit/{param}alpha/ for the last epoch .pt file
    in each condition.  Returns a list of:
        (label, bitdepth: int, alpha: float, last_ckpt: Path)
    sorted by bitdepth then alpha.
    """
    conditions = []
    for bd_dir in sorted(weights_dir.iterdir()):
        if not bd_dir.is_dir():
            continue
        try:
            bitdepth = int(bd_dir.name.replace("bit", ""))
        except ValueError:
            continue
        for param_dir in sorted(bd_dir.iterdir()):
            if not param_dir.is_dir():
                continue
            epochs = sorted(
                param_dir.glob("epoch*.pt"),
                key=lambda p: int(p.stem.replace("epoch", "")),
            )
            if not epochs:
                continue
            param_str = param_dir.name   # "0alpha", "0.2alpha", etc.
            alpha = float(param_str.replace("alpha", "")) if param_str != "0alpha" else 0.0
            label = f"{bitdepth}-bit  a={alpha}"
            conditions.append((label, bitdepth, alpha, epochs[-1]))
    conditions.sort(key=lambda c: (c[1], c[2]))
    return conditions


def load_model(ckpt_path: Path) -> CTCConv2DModel:
    model = CTCConv2DModel(
        n_mels=80, hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS, vocab_size=VOCAB_SIZE,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    return model


def collect_activations(model, loader, n_clips):
    """
    Run up to n_clips batches through model with forward hooks registered on
    model.conv and model.lstm.  Returns two flat numpy arrays:
        conv_vals : all activation values from the 2D CNN output
        lstm_vals : all activation values from the BiLSTM output
    """
    conv_buf, lstm_buf = [], []

    def _hook_conv(module, inp, out):
        # out: (B, 32, F=40, T')
        conv_buf.append(out.detach().cpu().numpy().ravel())

    def _hook_lstm(module, inp, out):
        # out: tuple (output (B, T', 256), (h_n, c_n))
        lstm_buf.append(out[0].detach().cpu().numpy().ravel())

    h1 = model.conv.register_forward_hook(_hook_conv)
    h2 = model.lstm.register_forward_hook(_hook_lstm)

    seen = 0
    with torch.no_grad():
        for feats, targets, raw_lengths, target_lengths in loader:
            if seen >= n_clips:
                break
            model(feats.to(DEVICE).transpose(1, 2))
            seen += feats.size(0)

    h1.remove()
    h2.remove()
    conv_vals = np.concatenate(conv_buf)
    lstm_vals = np.concatenate(lstm_buf)
    print(f"    [diag] conv  n={len(conv_vals):,}  "
          f"min={conv_vals.min():.3f}  max={conv_vals.max():.3f}  "
          f"mean={conv_vals.mean():.3f}  p95={np.percentile(conv_vals, 95):.3f}  "
          f"p99={np.percentile(conv_vals, 99):.3f}  "
          f"frac_zero={np.mean(conv_vals == 0):.3f}")
    print(f"    [diag] lstm  n={len(lstm_vals):,}  "
          f"min={lstm_vals.min():.3f}  max={lstm_vals.max():.3f}  "
          f"mean={lstm_vals.mean():.3f}")
    return conv_vals, lstm_vals


def compute_stats(values: np.ndarray) -> dict:
    return {
        "var":  float(np.var(values)),
        "std":  float(np.std(values)),
        "kurt": float(scipy_kurtosis(values, fisher=True)),  # excess kurtosis, Normal=0
        "mean": float(np.mean(values)),
    }


def kde_curve(values: np.ndarray, lo=None, hi_pct: float = 99.0,
              n_points: int = 512):
    """
    Return (x, density) for a KDE of values, subsampling to KDE_SAMPLES if needed.

    lo      : explicit lower bound; if None, uses the 1st percentile of the data.
    hi_pct  : upper percentile for the x-axis (default 99.0 -- clips the heavy tail).
    """
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
        # Degenerate distribution (e.g. all values identical after 1-bit quant)
        density = np.zeros_like(x)
    return x, density


# ── Plotting ──────────────────────────────────────────────────────────────────

def _cmap_for_condition(bitdepth, alpha):
    """Assign a colour: 1-bit → reds, 3-bit → blues, clean → black."""
    if bitdepth is None:
        return "black"
    n_alphas = 11
    t = alpha / 1.0  # normalise to [0,1]
    if bitdepth == 1:
        return plt.cm.Reds(0.4 + 0.5 * t)
    else:
        return plt.cm.Blues(0.4 + 0.5 * t)


def plot_pdfs(cross_stats, out_dir: Path):
    """
    Two figures: overlaid KDE curves at CNN output and LSTM output.
    cross_stats: list of (label, bd, alpha, conv_stats, lstm_stats, conv_vals, lstm_vals)

    CNN output (post-ReLU, all values >= 0):
      - Linear plot: lo=0, hi=95th percentile -- shows the main distribution body.
      - Log-x plot: lo=1e-3, full range -- reveals the heavy right tail structure.

    LSTM output (tanh-gated, bounded to (-1, 1)):
      - Standard linear plot; no special bounding needed.
    """
    # ── LSTM: standard linear plot ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    for label, bd, alpha, cs, ls, conv_vals, lstm_vals in cross_stats:
        color = _cmap_for_condition(bd, alpha)
        lsn   = "--" if bd is None else "-"
        x, dens = kde_curve(lstm_vals)   # default bounds fine for tanh output
        ax.plot(x, dens, color=color, linestyle=lsn, linewidth=1.5,
                label=label, alpha=0.85)
    ax.set_xlabel("Activation value")
    ax.set_ylabel("Density")
    ax.set_title(f"Empirical PDF — BiLSTM encoder output  (h_lstm)\n"
                 f"(tanh-gated, bounded ≈ (−1, 1)  |  "
                 f"ref model: {REFERENCE_CONDITION[0]}/{REFERENCE_CONDITION[1]})")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "pdf_lstm.png", dpi=150)
    plt.close(fig)
    print("  Saved pdf_lstm.png")

    # ── CNN: linear plot clipped at 95th percentile ───────────────────────────
    # Post-ReLU: all values >= 0.  The lo is pinned to 0 so the zero-mass from
    # ReLU is visible.  The tail is clipped at the 95th percentile to prevent
    # a few large outliers from compressing the interesting part of the plot.
    fig, ax = plt.subplots(figsize=(9, 4))
    for label, bd, alpha, cs, ls, conv_vals, lstm_vals in cross_stats:
        color = _cmap_for_condition(bd, alpha)
        lsn   = "--" if bd is None else "-"
        x, dens = kde_curve(conv_vals, lo=0.0, hi_pct=95.0)
        ax.plot(x, dens, color=color, linestyle=lsn, linewidth=1.5,
                label=label, alpha=0.85)
    # Find overall 95th percentile for axis label context
    all_conv = np.concatenate([s[5] for s in cross_stats])
    p95 = float(np.percentile(all_conv, 95))
    p99 = float(np.percentile(all_conv, 99))
    ax.set_xlabel(f"Activation value  (clipped at 95th pct ≈ {p95:.1f}; "
                  f"99th pct ≈ {p99:.1f})")
    ax.set_ylabel("Density")
    ax.set_title(f"Empirical PDF — CNN front-end output  (h_conv)  [linear scale]\n"
                 f"Post-ReLU (all values ≥ 0)  |  "
                 f"ref model: {REFERENCE_CONDITION[0]}/{REFERENCE_CONDITION[1]}")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "pdf_conv.png", dpi=150)
    plt.close(fig)
    print("  Saved pdf_conv.png")

    # ── CNN: log-x plot to expose the full right tail ─────────────────────────
    # Using log(x + ε) so the zero-mass is still visible at the far left.
    EPS = 1e-3
    fig, ax = plt.subplots(figsize=(9, 4))
    for label, bd, alpha, cs, ls, conv_vals, lstm_vals in cross_stats:
        color = _cmap_for_condition(bd, alpha)
        lsn   = "--" if bd is None else "-"
        log_vals = np.log10(conv_vals + EPS)
        x, dens = kde_curve(log_vals)
        ax.plot(x, dens, color=color, linestyle=lsn, linewidth=1.5,
                label=label, alpha=0.85)
    # Annotate tick labels with original scale
    ticks = ax.get_xticks()
    ax.set_xticklabels([f"{10**t - EPS:.2g}" for t in ticks])
    ax.set_xlabel("Activation value  (log₁₀ scale)")
    ax.set_ylabel("Density")
    ax.set_title(f"Empirical PDF — CNN front-end output  (h_conv)  [log scale]\n"
                 f"Shows full right tail structure  |  "
                 f"ref model: {REFERENCE_CONDITION[0]}/{REFERENCE_CONDITION[1]}")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "pdf_conv_log.png", dpi=150)
    plt.close(fig)
    print("  Saved pdf_conv_log.png")


def plot_stats_vs_alpha(matched_stats, out_dir: Path):
    """
    Variance and excess kurtosis vs. alpha, grouped by bitdepth.
    matched_stats: list of (label, bd, alpha, conv_stats, lstm_stats, conv_vals, lstm_vals)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for metric, col, ax in [("var", 3, axes[0]), ("kurt", 4, axes[1])]:
        metric_label = "Variance (σ²)" if metric == "var" else "Excess kurtosis"
        ax.set_xlabel("Dither alpha")
        ax.set_ylabel(metric_label)
        ax.set_title(f"{metric_label} vs. alpha  (matched eval)")
        ax.grid(True, alpha=0.3)

        for bitdepth in sorted({s[1] for s in matched_stats}):
            rows = sorted(
                [s for s in matched_stats if s[1] == bitdepth],
                key=lambda s: s[2],
            )
            alphas      = [r[2] for r in rows]
            conv_vals   = [r[3][metric] for r in rows]
            lstm_vals   = [r[4][metric] for r in rows]
            color       = "#d62728" if bitdepth == 1 else "#1f77b4"
            label_pref  = f"{bitdepth}-bit"
            ax.plot(alphas, conv_vals, "o-", color=color,
                    label=f"{label_pref} CNN",  linewidth=1.8, markersize=5)
            ax.plot(alphas, lstm_vals, "s--", color=color,
                    label=f"{label_pref} LSTM", linewidth=1.8, markersize=5,
                    alpha=0.65)
        ax.legend(fontsize=8)

    fig.suptitle("Activation distribution statistics — matched conditions\n"
                 "(each model evaluated on its own training quantization)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "stats_vs_alpha.png", dpi=150)
    plt.close(fig)
    print("  Saved stats_vs_alpha.png")


def plot_rel_var(cross_stats, out_dir: Path):
    """
    Variance normalised to the clean-audio condition, for CNN and LSTM layers.
    If the LSTM line stays near 1.0 while the CNN line varies, the network
    is filtering the quantisation noise.
    cross_stats: as above; first entry must be the clean condition.
    """
    clean_conv_var = cross_stats[0][3]["var"]
    clean_lstm_var = cross_stats[0][4]["var"]

    labels, rel_conv, rel_lstm = [], [], []
    for label, bd, alpha, cs, ls, _, _ in cross_stats:
        labels.append(label)
        rel_conv.append(cs["var"] / clean_conv_var if clean_conv_var > 0 else 0)
        rel_lstm.append(ls["var"] / clean_lstm_var if clean_lstm_var > 0 else 0)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 4))
    ax.axhline(1.0, color="k", linestyle=":", linewidth=1, label="clean baseline")
    ax.plot(x, rel_conv, "o-", color="#2ca02c", linewidth=1.8,
            markersize=6, label="CNN output (h_conv)")
    ax.plot(x, rel_lstm, "s--", color="#ff7f0e", linewidth=1.8,
            markersize=6, label="LSTM output (h_lstm)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Relative variance  (condition / clean)")
    ax.set_title("Filtering gain: variance relative to clean audio\n"
                 f"(reference model: {REFERENCE_CONDITION[0]}/{REFERENCE_CONDITION[1]})\n"
                 "LSTM line staying near 1.0 = network suppresses quantisation noise")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "rel_var.png", dpi=150)
    plt.close(fig)
    print("  Saved rel_var.png")


def save_summary_csv(matched_stats, cross_stats, out_dir: Path):
    path = out_dir / "summary.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "mode", "label", "bitdepth", "alpha",
            "conv_var", "conv_kurtosis",
            "lstm_var", "lstm_kurtosis",
        ])
        for label, bd, alpha, cs, ls, _, _ in matched_stats:
            writer.writerow([
                "matched", label, bd, alpha,
                f"{cs['var']:.6f}", f"{cs['kurt']:.4f}",
                f"{ls['var']:.6f}", f"{ls['kurt']:.4f}",
            ])
        for label, bd, alpha, cs, ls, _, _ in cross_stats:
            writer.writerow([
                "cross", label, bd if bd else "clean", alpha if alpha is not None else "",
                f"{cs['var']:.6f}", f"{cs['kurt']:.4f}",
                f"{ls['var']:.6f}", f"{ls['kurt']:.4f}",
            ])
    print(f"  Saved summary.csv")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    conditions = discover_conditions(WEIGHTS_DIR)
    if not conditions:
        raise RuntimeError(f"No weight files found under {WEIGHTS_DIR}")
    print(f"Found {len(conditions)} trained conditions:")
    for label, bd, alpha, ckpt in conditions:
        print(f"  {label:22s}  last epoch: {ckpt.stem}")

    # Shared validation dataset; quantize_fn is swapped per-condition below
    val_ds = SpeechCommandsCTC(DATA_ROOT, subset="validation", quantize_fn=None)
    rng = random.Random(RNG_SEED)
    indices = list(range(len(val_ds)))
    rng.shuffle(indices)
    val_subset = Subset(val_ds, indices[:N_CLIPS])

    def make_loader():
        return DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False,
                          collate_fn=ctc_collate, num_workers=0)

    # ── 1. MATCHED: each model on its own training condition ──────────────────
    print("\n-- MATCHED analysis ---------------------------------------------")
    matched_stats = []
    for label, bitdepth, alpha, ckpt in conditions:
        print(f"  {label}  [{ckpt.stem}]")
        model = load_model(ckpt)
        val_ds.quantize_fn = make_quantize_fn(bitdepth, alpha)
        conv_vals, lstm_vals = collect_activations(model, make_loader(), N_CLIPS)
        cs, ls = compute_stats(conv_vals), compute_stats(lstm_vals)
        print(f"    conv  var={cs['var']:.5f}  kurt={cs['kurt']:+.3f}")
        print(f"    lstm  var={ls['var']:.5f}  kurt={ls['kurt']:+.3f}")
        matched_stats.append((label, bitdepth, alpha, cs, ls, conv_vals, lstm_vals))

    # ── 2. CROSS: one reference model, all conditions + clean ─────────────────
    ref_dir = WEIGHTS_DIR / REFERENCE_CONDITION[0] / REFERENCE_CONDITION[1]
    ref_ckpts = sorted(ref_dir.glob("epoch*.pt"),
                       key=lambda p: int(p.stem.replace("epoch", "")))
    if not ref_ckpts:
        raise RuntimeError(f"No checkpoints in {ref_dir}")
    ref_model = load_model(ref_ckpts[-1])
    print(f"\n-- CROSS analysis  (reference: {ref_ckpts[-1]}) ------------------")

    cross_inputs = [("clean  (no quant)", None, None)] + \
                   [(lbl, bd, al) for lbl, bd, al, *_ in conditions]

    cross_stats = []
    for label, bitdepth, alpha in cross_inputs:
        print(f"  {label}")
        val_ds.quantize_fn = make_quantize_fn(bitdepth, alpha)
        conv_vals, lstm_vals = collect_activations(ref_model, make_loader(), N_CLIPS)
        cs, ls = compute_stats(conv_vals), compute_stats(lstm_vals)
        print(f"    conv  var={cs['var']:.5f}  kurt={cs['kurt']:+.3f}")
        print(f"    lstm  var={ls['var']:.5f}  kurt={ls['kurt']:+.3f}")
        cross_stats.append((label, bitdepth, alpha, cs, ls, conv_vals, lstm_vals))

    # ── Plots ──────────────────────────────────────────────────────────────────
    print("\n-- Generating plots ----------------------------------------------")
    plot_pdfs(cross_stats, OUT_DIR)
    plot_stats_vs_alpha(matched_stats, OUT_DIR)
    plot_rel_var(cross_stats, OUT_DIR)
    save_summary_csv(matched_stats, cross_stats, OUT_DIR)

    # Print summary table
    print(f"\n{'Condition':<25} {'Mode':<8} {'Conv var':>10} {'Conv kurt':>11} "
          f"{'LSTM var':>10} {'LSTM kurt':>11}")
    print("-" * 78)
    for lbl, bd, al, cs, ls, _, _ in matched_stats:
        print(f"{lbl:<25} {'matched':<8} {cs['var']:>10.5f} {cs['kurt']:>+11.3f} "
              f"{ls['var']:>10.5f} {ls['kurt']:>+11.3f}")

    print(f"\nAll outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
