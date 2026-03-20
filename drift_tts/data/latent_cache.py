"""Pre-encode pipeline: audio → DACVAE latent + UTF-8 tokens → cached .pt files.

Usage:
    python scripts/cache_latents.py --manifest train.jsonl --audio_root /data/emilia/audio --output /data/emilia/cache
"""

import json
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm


def cache_latents(
    manifest_path: str,
    audio_root: str,
    output_root: str,
    dacvae_model_path: str = "facebook/dacvae-watermarked",
    chunk_duration: float = 10.0,
    device: str = "cuda",
    max_entries: Optional[int] = None,
) -> None:
    """Pre-encode the entire dataset to DACVAE latents.

    Writes one .pt file per chunk containing:
        {"latent": [T, D], "tokens": [L], "speaker_cluster": int}
    """
    from drift_tts.codec.dacvae import DACVAECodec
    from drift_tts.data.audio_processing import load_and_resample, chunk_waveform
    from drift_tts.data.text_processing import UTF8Tokenizer

    output = Path(output_root) / "latents"
    output.mkdir(parents=True, exist_ok=True)

    dacvae = DACVAECodec.load(dacvae_model_path, device=device)
    tokenizer = UTF8Tokenizer()

    chunk_samples = int(chunk_duration * dacvae.sample_rate)

    entries = []
    with open(manifest_path) as f:
        for line in f:
            entries.append(json.loads(line.strip()))
    if max_entries:
        entries = entries[:max_entries]

    idx = 0
    for entry in tqdm(entries, desc="Caching latents"):
        audio_path = Path(audio_root) / entry["audio_path"]
        if not audio_path.exists():
            continue

        waveform = load_and_resample(str(audio_path), dacvae.sample_rate)
        chunks = chunk_waveform(waveform, chunk_samples=chunk_samples)

        text = entry.get("text", "")
        token_ids = tokenizer.encode(text)
        speaker_cluster = entry.get("speaker_cluster", entry.get("speaker_id", 0))

        for chunk in chunks:
            chunk = chunk.unsqueeze(0).to(device)  # [1, 1, T]
            latent = dacvae.encode(chunk)  # [1, T_latent, D]
            latent = latent.squeeze(0).cpu()  # [T_latent, D]

            torch.save(
                {
                    "latent": latent,
                    "tokens": torch.tensor(token_ids, dtype=torch.long),
                    "speaker_cluster": int(speaker_cluster),
                },
                output / f"{idx:08d}.pt",
            )
            idx += 1

    print(f"Cached {idx} latent chunks to {output}")
