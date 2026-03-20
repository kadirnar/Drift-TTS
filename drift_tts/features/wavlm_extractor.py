"""WavLM multi-scale feature extractor for drift loss.

Mirrors the multi-scale activation pattern in drifting/models/mae_model.py
``get_activations``, but for audio using WavLM-Large (24 transformer layers).

Layer groups (multi-radius analogue):
  "acoustic":      layers  1-6   (low-level acoustic)
  "phonetic_low":  layers  7-12  (phonetic structure)
  "phonetic_high": layers 13-18  (linguistic)
  "semantic":      layers 19-24  (meaning + speaker)

Each group produces mean and std statistics, matching the original
patch_mean_size / patch_std_size feature aggregation.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torchaudio


# WavLM operates at 16 kHz
_WAVLM_SR = 16_000

# Layer group definitions (0-indexed hidden states: 0 = CNN output, 1-24 = transformer layers)
_LAYER_GROUPS: Dict[str, Tuple[int, int]] = {
    "acoustic": (1, 6),
    "phonetic_low": (7, 12),
    "phonetic_high": (13, 18),
    "semantic": (19, 24),
}


def _safe_std(x: torch.Tensor, dim: int, eps: float = 1e-6) -> torch.Tensor:
    x32 = x.float()
    mean = x32.mean(dim=dim, keepdim=True)
    var = ((x32 - mean) ** 2).mean(dim=dim, keepdim=False)
    return torch.sqrt(var.clamp(min=0.0) + eps)


class WavLMFeatureExtractor(nn.Module):
    """Frozen WavLM-Large producing multi-scale features for drift loss.

    The model is entirely frozen; gradients flow through the decoder
    waveform into WavLM to supervise the generator.
    """

    def __init__(
        self,
        model_name: str = "microsoft/wavlm-large",
        source_sr: int = 44_100,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.source_sr = source_sr
        self.use_gradient_checkpointing = use_gradient_checkpointing

        from transformers import WavLMModel

        self._wavlm = WavLMModel.from_pretrained(model_name)
        self._wavlm.eval()
        for p in self._wavlm.parameters():
            p.requires_grad_(False)
        if use_gradient_checkpointing:
            self._wavlm.gradient_checkpointing_enable()

        self._resampler: Optional[nn.Module] = None

    def _get_resampler(self, device: torch.device) -> torchaudio.transforms.Resample:
        if self._resampler is None:
            self._resampler = torchaudio.transforms.Resample(
                self.source_sr, _WAVLM_SR
            ).to(device)
        return self._resampler

    def extract_hidden_states(
        self, waveform: torch.Tensor
    ) -> List[torch.Tensor]:
        """Run WavLM and return all hidden states.

        Args:
            waveform: [B, T_audio] at source_sr.

        Returns:
            List of 25 tensors, each [B, T_wav, 1024].
        """
        # Resample to 16 kHz
        if self.source_sr != _WAVLM_SR:
            resampler = self._get_resampler(waveform.device)
            waveform = resampler(waveform)

        outputs = self._wavlm(
            waveform,
            output_hidden_states=True,
            return_dict=True,
        )
        return list(outputs.hidden_states)

    def forward(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract multi-scale features.

        Args:
            waveform: [B, 1, T_audio] at source_sr (mono). Squeezed internally.

        Returns:
            Dict of feature tensors shaped [B, T, D] or [B, 1, D].
        """
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)

        hidden_states = self.extract_hidden_states(waveform)
        features: Dict[str, torch.Tensor] = {}

        for group_name, (start, end) in _LAYER_GROUPS.items():
            # Stack layers in group: [B, T, D, num_layers]
            group_stack = torch.stack(hidden_states[start : end + 1], dim=-1)
            # Average across layers → [B, T, D]
            group_feat = group_stack.mean(dim=-1)

            features[group_name] = group_feat  # [B, T, D]
            features[f"{group_name}_mean"] = group_feat.mean(dim=1, keepdim=True)  # [B, 1, D]
            features[f"{group_name}_std"] = _safe_std(group_feat, dim=1).unsqueeze(1)  # [B, 1, D]

        # Global feature (last hidden state, pooled)
        last_hidden = hidden_states[-1]
        features["global_mean"] = last_hidden.mean(dim=1, keepdim=True)
        features["global_std"] = _safe_std(last_hidden, dim=1).unsqueeze(1)

        return features
