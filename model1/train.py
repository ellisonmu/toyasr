"""
train1.py -- waveform -> 1D-conv front end -> BiLSTM + CTC
"""

import argparse
import csv
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

_toymodel = Path(__file__).parent.parent   # python-files/toymodel/
_root     = _toymodel.parent               # python-files/
sys.path.insert(0, str(_toymodel))                  # model.py, dataset.py, plot_sweep.py
sys.path.insert(0, str(_root))                      # from experiments.X import Y
sys.path.insert(0, str(_root / "experiments"))      # bare imports inside dithering.py

from model      import CTCConv1DModel
from dataset    import WaveformCommandsCTC, waveform_ctc_collate, decode, BLANK_IDX, VOCAB_SIZE
from plot_sweep import run as plot_sweep
from experiments.quantization import uniform_quantization
from experiments.dithering    import dither_app

DATA_ROOT   = _toymodel / "data"                   # toymodel/data/
RESULTS_DIR = Path(__file__).parent / "results"    # toymodel/model1/results/
WEIGHTS_DIR = Path(__file__).parent / "weights"    # toymodel/model1/weights/
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BITDEPTH = 3
PARAMS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
HIDDEN_DIM = 128
NUM_LAYERS = 2
EPOCHS     = 30
BATCH_SIZE = 64
LR         = 1e-3
WARMUP_STEPS = 200

def alpha_str(p):
    """Format a sweep parameter for directory names: 0.0 -> '0', 1.0 -> '1', else '0.x'."""
    return str(int(p)) if p == int(p) else str(p)

def evaluate(model, loader):
    """Greedy CTC decode (collapse repeats, drop blanks) -> WER/CER vs label."""
    model.eval()
    refs, hyps = [], []
    with torch.no_grad():
        for waveforms, targets, raw_lengths, target_lengths in loader:
            waveforms = waveforms.to(DEVICE)
            log_probs = model(waveforms) # (B, T', V)
            input_lengths = model.output_length(raw_lengths) # raw samples -> conv frames
            preds = log_probs.argmax(dim=-1).cpu() # (B, T')

            offset = 0
            for b in range(waveforms.size(0)):
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
    global BITDEPTH
    parser = argparse.ArgumentParser()
    parser.add_argument("--bitdepth", type=int, default=BITDEPTH)
    args = parser.parse_args()
    BITDEPTH = args.bitdepth

    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    train_ds = WaveformCommandsCTC(DATA_ROOT, subset="training",   quantize_fn=None)
    val_ds   = WaveformCommandsCTC(DATA_ROOT, subset="validation", quantize_fn=None)
    print(f"train: {len(train_ds)}  val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=waveform_ctc_collate, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=waveform_ctc_collate, num_workers=4)

    for param in PARAMS:
        quantize_fn = partial(dither_app, bitdepth=BITDEPTH, param=param,
                              Qtype="uniform", Dtype="subtractive")
        train_ds.quantize_fn = quantize_fn
        val_ds.quantize_fn   = quantize_fn

        run_dir     = RESULTS_DIR / f"{BITDEPTH}bit" / f"{alpha_str(param)}alpha"
        weights_dir = WEIGHTS_DIR  / f"{BITDEPTH}bit" / f"{alpha_str(param)}alpha"
        run_dir.mkdir(parents=True, exist_ok=True)
        weights_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== bitdepth={BITDEPTH}  alpha={param} ===")

        model = CTCConv1DModel(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
                               vocab_size=VOCAB_SIZE).to(DEVICE)
        nn.init.constant_(model.fc.bias[BLANK_IDX], -2.0)
        criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        wer_history, cer_history = [], []
        global_step = 0

        for epoch in range(1, EPOCHS + 1):
            model.train()
            running_loss = 0.0
            for waveforms, targets, raw_lengths, target_lengths in train_loader:
                # Linear LR warmup over the first WARMUP_STEPS steps, ramping
                # 0 -> LR, to avoid the model collapsing to all-blank before
                # it sees enough gradient signal. Holds at LR afterward.
                warmup_lr = LR * min(1.0, (global_step + 1) / WARMUP_STEPS)
                for group in optimizer.param_groups:
                    group["lr"] = warmup_lr

                waveforms, targets = waveforms.to(DEVICE), targets.to(DEVICE)
                log_probs = model(waveforms)                       # (B, T', V)
                input_lengths = model.output_length(raw_lengths)   # raw samples -> conv frames
                loss = criterion(
                    log_probs.transpose(0, 1),                     # CTCLoss wants (T', B, V)
                    targets, input_lengths, target_lengths,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                global_step += 1

            metrics = evaluate(model, val_loader)
            wer_history.append(metrics["wer"])
            cer_history.append(metrics["cer"])
            print(f"epoch {epoch:2d}  loss {running_loss / len(train_loader):.4f}  "
                  f"val WER {metrics['wer']:.3f}  val CER {metrics['cer']:.3f}")
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
        plt.title(f"Validation WER/CER vs Epoch (bitdepth={BITDEPTH}, param={param})")
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)
        plt.savefig(run_dir / "wer_cer_vs_epoch.png")
        plt.close()

        print(f"Done param={param}. Results -> {run_dir}/  Weights -> {weights_dir}/")

    print(f"\nAll done. Results under {RESULTS_DIR}/{BITDEPTH}bit/")
    plot_sweep(RESULTS_DIR, BITDEPTH)

if __name__ == "__main__":
    main()
