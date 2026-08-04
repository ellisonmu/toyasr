"""
generate_opus_corpus.py
Pre-compresses Google Speech Commands (v0.02) through the Opus codec once
"""

import csv
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import soundfile as sf
import torchaudio

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))
from experiments.opus_codec import opus_quantize

SR = 16_000
BITRATE = 16_000   # bps - sweep

# output 
DATA_ROOT   = Path(__file__).parent / "data"    
OUT_DIR     = Path(__file__).parent / f"opus_corpus_{BITRATE // 1000}kbps"
SUBSETS     = ["training", "validation", "testing"]
MAX_SAMPLES = None    
WORKERS     = 8   

def compress_one(ds, idx, audio_dir):
    """Round-trip one clip through Opus and cache it to disk (resumable)."""
    waveform, sample_rate, label, speaker_id, utterance_number = ds[idx]
    assert sample_rate == SR, f"expected {SR} Hz, got {sample_rate}"
    out_path = audio_dir / f"{label}_{speaker_id}_{utterance_number:02d}.wav"
    if not out_path.exists():
        audio = waveform.squeeze(0).numpy().astype(np.float32)
        compressed = opus_quantize(audio, bitrate=BITRATE, sr=SR)
        sf.write(str(out_path), compressed, SR, subtype="FLOAT")
    return str(out_path), label


def main():
    print(f"Codec  : Opus @ {BITRATE} bps")
    print(f"Output : {OUT_DIR}\n")

    for subset in SUBSETS:
        ds = torchaudio.datasets.SPEECHCOMMANDS(
            root=str(DATA_ROOT), download=True, subset=subset,
        )
        n = len(ds) if MAX_SAMPLES is None else min(MAX_SAMPLES, len(ds))
        print(f"[{subset}] {n} clip(s)")

        audio_dir = OUT_DIR / subset
        audio_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = OUT_DIR / f"{subset}_manifest.csv"

        results = []
        done = 0
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {pool.submit(compress_one, ds, i, audio_dir): i for i in range(n)}
            for fut in as_completed(futures):
                results.append(fut.result())
                done += 1
                if done % 200 == 0 or done == n:
                    print(f"  {done}/{n} done")

        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["audio_file", "label"])
            writer.writeheader()
            for audio_file, label in results:
                writer.writerow({"audio_file": audio_file, "label": label})
        print(f"  Manifest: {manifest_path}\n")

    print("Done. Load with, e.g.:")
    print("    ManifestCommandsCTC(out_dir / 'training_manifest.csv',")
    print("                        feature_type='waveform', quantize_fn=None)")


if __name__ == "__main__":
    main()
