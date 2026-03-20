"""Evaluation script: UTMOS, WER, Speaker Similarity, RTF."""

import argparse
import time
from pathlib import Path

import torch
import torchaudio


def compute_rtf(tts, text, ref_audio, num_runs=5):
    """Measure real-time factor."""
    # Warmup
    tts.generate(text, ref_audio)

    durations = []
    for _ in range(num_runs):
        start = time.time()
        waveform = tts.generate(text, ref_audio)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        audio_duration = waveform.shape[-1] / 44100.0
        durations.append(elapsed / audio_duration)
    return sum(durations) / len(durations)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Drift-TTS")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_manifest", type=str, required=True, help="JSONL with test utterances.")
    parser.add_argument("--output_dir", type=str, default="eval_outputs")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples", type=int, default=100)
    args = parser.parse_args()

    from drift_tts.inference.generate import DriftTTSInference

    import json

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tts = DriftTTSInference(checkpoint_path=args.checkpoint, device=args.device)

    entries = []
    with open(args.test_manifest) as f:
        for line in f:
            entries.append(json.loads(line.strip()))
    entries = entries[: args.max_samples]

    results = []
    for i, entry in enumerate(entries):
        text = entry["text"]
        ref_audio = entry.get("ref_audio", entry.get("audio_path"))

        waveform = tts.generate(text, ref_audio)
        out_path = output_dir / f"{i:04d}.wav"
        tts.save_audio(waveform, str(out_path))

        results.append({
            "idx": i,
            "text": text,
            "output": str(out_path),
            "duration": waveform.shape[-1] / 44100.0,
        })

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Compute RTF on first sample
    if entries:
        rtf = compute_rtf(tts, entries[0]["text"], entries[0].get("ref_audio", entries[0].get("audio_path")))
        print(f"RTF: {rtf:.4f}")

    print(f"Generated {len(results)} samples in {output_dir}")
    print("Run UTMOS / WER / Speaker Similarity externally on the generated samples.")


if __name__ == "__main__":
    main()
