"""
train2.py -- waveform -> log-mel spectrogram -> 2D-conv front end -> BiLSTM + CTC
"""

import argparse
import csv
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import jiwer

torch.backends.cudnn.benchmark = True

_toymodel = Path(__file__).parent.parent   # python-files/toymodel/
_root     = _toymodel.parent               # python-files/
sys.path.insert(0, str(_toymodel))                  # model.py, dataset.py, plot_sweep.py
sys.path.insert(0, str(_root))                      # from experiments.X import Y
sys.path.insert(0, str(_root / "experiments"))      # bare imports inside dithering.py

from model      import CTCConv2DModel
from dataset    import SpeechCommandsCTC, ctc_collate, decode, BLANK_IDX, VOCAB_SIZE
from plot_sweep import run as plot_sweep
from experiments.dithering import dither_app

DATA_ROOT   = _toymodel / "data"                   # toymodel/data/
RESULTS_DIR = Path(__file__).parent / "results"    # toymodel/model2/results/
WEIGHTS_DIR = Path(__file__).parent / "weights"    # toymodel/model2/weights/
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BITDEPTH = 2
PARAMS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

HIDDEN_DIM   = 128
NUM_LAYERS   = 2
EPOCHS       = 30
BATCH_SIZE   = 64
LR           = 1e-3
WARMUP_STEPS = 200
NUM_WORKERS  = min(4, os.cpu_count() or 1)

def alpha_str(p):
    """Format a sweep parameter for directory names: 0.0 -> '0', 1.0 -> '1', else '0.x'."""
    return str(int(p)) if p == int(p) else str(p)

def evaluate(model, loader, use_amp=False):
    """Greedy CTC decode (collapse repeats, drop blanks) -> WER/CER vs label."""
    model.eval()
    refs, hyps = [], []
    with torch.no_grad():
        for feats, targets, raw_lengths, target_lengths in loader:
            feats = feats.to(DEVICE, non_blocking=True)
            with torch.autocast(device_type=DEVICE.type, enabled=use_amp):
                log_probs = model(feats.transpose(1, 2))          # (B, n_mels, T) -> (B, T', V)
            input_lengths = model.output_length(raw_lengths)  # mel frames -> conv frames
            preds = log_probs.float().argmax(dim=-1).cpu()    # (B, T')

            offset = 0
            for b in range(feats.size(0)):
                t_len = target_lengths[b].item()
                ref = decode(targets[offset:offset + t_len].tolist())
                offset += t_len

                pred = preds[b, : input_lengths[b]].tolist()
                collapsed, prev = [], None
                for p in pred:
                    if p != prev and p != BLANK_IDX:
                        collapsed.append(p)
                    prev = p
                hyp = decode(collapsed)

                refs.append(ref if ref else " ")
                hyps.append(hyp if hyp else " ")
    return {"wer": jiwer.wer(refs, hyps), "cer": jiwer.cer(refs, hyps)}

def main():
    global BITDEPTH, PARAMS
    parser = argparse.ArgumentParser()
    parser.add_argument("--bitdepth", type=int, default=BITDEPTH)
    parser.add_argument(
        "--no-quant", action="store_true",
        help="Run one condition with clean audio (no quantization). "
             "Use this to verify the model can learn before running the dither sweep.",
    )
    args = parser.parse_args()
    BITDEPTH = args.bitdepth
    if args.no_quant:
        PARAMS = [None]   # None → quantize_fn=None → clean audio

    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    use_amp = DEVICE.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    train_ds = SpeechCommandsCTC(DATA_ROOT, subset="training",   quantize_fn=None)
    val_ds   = SpeechCommandsCTC(DATA_ROOT, subset="validation", quantize_fn=None)
    print(f"train: {len(train_ds)}  val: {len(val_ds)}")

    for param in PARAMS:
        if param is None:
            quantize_fn = None
            param_label = "clean"
        else:
            quantize_fn = partial(dither_app, bitdepth=BITDEPTH, param=param,
                                  Qtype="uniform", Dtype="subtractive")
            param_label = f"{alpha_str(param)}alpha"
        train_ds.quantize_fn = quantize_fn
        val_ds.quantize_fn   = quantize_fn

        # DataLoaders are recreated per param so worker processes inherit the
        # correct quantize_fn; worker state is snapshotted at construction time,
        # so mutating the dataset attribute after creation would be invisible
        # to workers when num_workers > 0.
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True,
            collate_fn=ctc_collate, num_workers=NUM_WORKERS,
            pin_memory=use_amp, persistent_workers=(NUM_WORKERS > 0),
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False,
            collate_fn=ctc_collate, num_workers=NUM_WORKERS,
            pin_memory=use_amp, persistent_workers=(NUM_WORKERS > 0),
        )

        run_dir     = RESULTS_DIR / f"{BITDEPTH}bit" / param_label
        weights_dir = WEIGHTS_DIR  / f"{BITDEPTH}bit" / param_label
        run_dir.mkdir(parents=True, exist_ok=True)
        weights_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== bitdepth={BITDEPTH}  alpha={param} ===")

        model = CTCConv2DModel(n_mels=80, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
                               vocab_size=VOCAB_SIZE).to(DEVICE)
        # Bias blank slightly negative so the model doesn't collapse to all-blank early.
        nn.init.constant_(model.fc.bias[BLANK_IDX], -2.0)
        criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        wer_history, cer_history = [], []
        global_step = 0

        for epoch in range(1, EPOCHS + 1):
            model.train()
            running_loss = 0.0
            for feats, targets, raw_lengths, target_lengths in train_loader:
                # Linear LR warmup over the first WARMUP_STEPS steps, ramping
                # 0 -> LR, to avoid the model collapsing to all-blank before
                # it sees enough gradient signal. CosineAnnealingLR (stepped
                # once per epoch, below) takes over once warmup completes.
                if global_step < WARMUP_STEPS:
                    warmup_lr = LR * (global_step + 1) / WARMUP_STEPS
                    for group in optimizer.param_groups:
                        group["lr"] = warmup_lr

                feats   = feats.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                # feats: (B, frames, n_mels) from ctc_collate -> model wants (B, n_mels, frames)
                with torch.autocast(device_type=DEVICE.type, enabled=use_amp):
                    log_probs = model(feats.transpose(1, 2))           # (B, T', V)
                    input_lengths = model.output_length(raw_lengths)   # mel frames -> conv frames
                    loss = criterion(
                        log_probs.transpose(0, 1),                     # CTCLoss wants (T', B, V)
                        targets, input_lengths, target_lengths,
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()
                global_step += 1

            scheduler.step()
            metrics = evaluate(model, val_loader, use_amp=use_amp)
            wer_history.append(metrics["wer"])
            cer_history.append(metrics["cer"])
            print(f"epoch {epoch:2d}  loss {running_loss / len(train_loader):.4f}  "
                  f"val WER {metrics['wer']:.3f}  val CER {metrics['cer']:.3f}  "
                  f"lr {scheduler.get_last_lr()[0]:.2e}")
            torch.save(model.state_dict(), weights_dir / f"epoch{epoch}.pt")

        with open(run_dir / "wer_cer_vs_epoch.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "wer", "cer"])
            for e, (w, c) in enumerate(zip(wer_history, cer_history), start=1):
                writer.writerow([e, w, c])

        epochs_range = list(range(1, EPOCHS + 1))
        plt.figure()
        plt.plot(epochs_range, wer_history, marker="o", label="WER")
        plt.plot(epochs_range, cer_history, marker="o", label="CER")
        plt.xlabel("Epoch")
        plt.ylabel("Error rate")
        plt.title(f"Validation WER/CER vs Epoch (bitdepth={BITDEPTH}, {param_label})")
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)
        plt.savefig(run_dir / "wer_cer_vs_epoch.png")
        plt.close()

        print(f"Done {param_label}. Results -> {run_dir}/  Weights -> {weights_dir}/")

    print(f"\nAll done. Results under {RESULTS_DIR}/{BITDEPTH}bit/")
    plot_sweep(RESULTS_DIR, BITDEPTH)

if __name__ == "__main__":
    main()
