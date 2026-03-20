"""Emilia dataset loader for Drift-TTS.

Supports two modes:
1. Raw audio mode: loads audio + text, encodes on-the-fly.
2. Cached latent mode: loads pre-encoded DACVAE latents + BPE tokens.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader


class EmiliaDataset(Dataset):
    """Emilia TTS dataset.

    Expected directory structure:
        data_root/
            manifests/
                train.jsonl   # {"audio_path": ..., "text": ..., "speaker_id": ..., "duration": ...}
            audio/
                ...wav files...
    """

    def __init__(
        self,
        manifest_path: str,
        audio_root: str,
        chunk_duration: float = 10.0,
        sample_rate: int = 44_100,
        max_text_length: int = 512,
        tokenizer=None,
    ):
        self.audio_root = Path(audio_root)
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.max_text_length = max_text_length
        self.tokenizer = tokenizer

        self.entries = []
        with open(manifest_path) as f:
            for line in f:
                entry = json.loads(line.strip())
                self.entries.append(entry)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        entry = self.entries[idx]
        from drift_tts.data.audio_processing import load_and_resample, pad_or_trim, CHUNK_SAMPLES

        # Load audio
        audio_path = self.audio_root / entry["audio_path"]
        waveform = load_and_resample(str(audio_path), self.sample_rate)
        waveform, audio_len = pad_or_trim(waveform, CHUNK_SAMPLES)

        # Tokenize text
        text = entry.get("text", "")
        if self.tokenizer is not None:
            token_ids = self.tokenizer.encode(text)
            token_ids = token_ids[: self.max_text_length]
        else:
            token_ids = [0]

        speaker_id = entry.get("speaker_cluster", entry.get("speaker_id", 0))

        return {
            "waveform": waveform.squeeze(0),  # [T_audio]
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "speaker_id": speaker_id,
            "audio_len": audio_len,
        }


class CachedLatentDataset(Dataset):
    """Dataset loading pre-encoded DACVAE latents and BPE tokens.

    Expected directory structure:
        cache_root/
            latents/
                000000.pt  # {"latent": [1024, T], "tokens": [L], "speaker_cluster": int}
                000001.pt
                ...
    """

    def __init__(
        self,
        cache_root: str,
        target_frames: int = 862,
        max_text_length: int = 512,
    ):
        self.cache_root = Path(cache_root)
        self.target_frames = target_frames
        self.max_text_length = max_text_length
        self.files = sorted((self.cache_root / "latents").glob("*.pt"))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = torch.load(self.files[idx], map_location="cpu", weights_only=True)
        latent = data["latent"]  # [T, D] channels last
        tokens = data["tokens"]  # [L]
        speaker_cluster = data.get("speaker_cluster", 0)

        # Pad/trim latent along time axis (dim 0)
        T = latent.shape[0]
        if T < self.target_frames:
            latent = torch.nn.functional.pad(latent, (0, 0, 0, self.target_frames - T))
        elif T > self.target_frames:
            latent = latent[: self.target_frames]

        # Truncate tokens
        tokens = tokens[: self.max_text_length]

        return {
            "latent": latent,
            "tokens": tokens,
            "speaker_cluster": speaker_cluster,
            "num_frames": min(T, self.target_frames),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for CachedLatentDataset with variable-length tokens."""
    latents = torch.stack([b["latent"] for b in batch])
    speaker_clusters = torch.tensor([b["speaker_cluster"] for b in batch], dtype=torch.long)
    num_frames = torch.tensor([b["num_frames"] for b in batch], dtype=torch.long)

    # Pad tokens to max length in batch
    max_len = max(b["tokens"].shape[0] for b in batch)
    token_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    token_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, b in enumerate(batch):
        L = b["tokens"].shape[0]
        token_ids[i, :L] = b["tokens"]
        token_mask[i, :L] = 1

    return {
        "latent": latents,
        "token_ids": token_ids,
        "token_mask": token_mask,
        "speaker_cluster": speaker_clusters,
        "num_frames": num_frames,
    }


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 8,
    shuffle: bool = True,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a DataLoader with appropriate collate function."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn if isinstance(dataset, CachedLatentDataset) else None,
        drop_last=True,
    )


def infinite_sampler(loader: DataLoader):
    """Infinite iterator over a DataLoader."""
    while True:
        for batch in loader:
            yield batch
