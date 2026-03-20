"""Audio preprocessing: resample, chunk, pad.

Sample rate is determined by the DACVAE model (not hardcoded).
"""

from typing import Optional, Tuple

import torch
import torchaudio


def load_and_resample(path: str, target_sr: int) -> torch.Tensor:
    """Load audio file and resample to target_sr mono.

    Returns:
        [1, T] waveform tensor.
    """
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform


def chunk_waveform(
    waveform: torch.Tensor,
    chunk_samples: int,
    hop_samples: Optional[int] = None,
) -> list:
    """Split waveform into fixed-length chunks.

    Args:
        waveform: [1, T] or [T] waveform.
        chunk_samples: samples per chunk.
        hop_samples: stride between chunks (default: same as chunk_samples).

    Returns:
        List of [1, chunk_samples] tensors.
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    T = waveform.shape[-1]
    if hop_samples is None:
        hop_samples = chunk_samples

    chunks = []
    for start in range(0, T, hop_samples):
        end = start + chunk_samples
        if end > T:
            chunk = torch.zeros(1, chunk_samples, dtype=waveform.dtype)
            chunk[:, : T - start] = waveform[:, start:]
        else:
            chunk = waveform[:, start:end]
        chunks.append(chunk)
    return chunks


def pad_or_trim(
    waveform: torch.Tensor,
    target_samples: int,
) -> Tuple[torch.Tensor, int]:
    """Pad or trim waveform to target length.

    Returns:
        (padded_waveform, original_length)
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    T = waveform.shape[-1]
    if T >= target_samples:
        return waveform[:, :target_samples], min(T, target_samples)
    padded = torch.zeros(1, target_samples, dtype=waveform.dtype)
    padded[:, :T] = waveform
    return padded, T
