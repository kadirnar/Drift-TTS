# Drift-TTS

One-step real-time text-to-speech via generative drifting.

Drift-TTS applies the [Drift](https://arxiv.org/abs/2602.04770) generative paradigm to speech synthesis. The Drift model moves the entire iterative denoising process of diffusion/flow models into training time, enabling **single-step generation** at inference. This is the first application of Drift to TTS.

## Architecture

```
TRAINING:
  Text  → UTF-8 Bytes → Text Transformer → Text Features ───────┐
  Ref Audio → DACVAE Encode → Speaker Encoder → Speaker Embed ───┤ (AdaLN + Cross-Attn)
  Noise ~ N(0,I) → AudioDiTGen → Predicted Latent [B, 1024, T]
  Real Audio → DACVAE Encode → Target Latent [B, 1024, T]

  Drift Loss (WavLM feature space):
    Predicted → DACVAE Decode → Waveform → WavLM → Gen Features
    Memory Bank → WavLM → Pos/Neg Features
    Loss = drift_loss(gen, pos, neg)

INFERENCE (1 step):
  Text + Ref Audio → AudioDiTGen → Latent → DACVAE Decode → Waveform
```

### Key Components

| Component | Details |
|-----------|---------|
| **Audio Codec** | DACVAE (`facebook/dacvae-watermarked`), 44.1kHz, 1024-dim latent, 512x compression |
| **Text Encoding** | UTF-8 byte-level transformer (Echo TTS style), vocab=259, bidirectional |
| **Voice Cloning** | Zero-shot from ~3-10s reference audio via dual-path speaker encoder |
| **Generator** | AudioDiTGen: 22-layer DiT with self-attn + cross-attn + SwiGLU FFN + AdaLN |
| **Feature Extractor** | WavLM-Large multi-scale features (training only) |
| **Loss** | Drift loss with multi-radius temperatures R=[0.02, 0.05, 0.2] |
| **CFG** | Dual classifier-free guidance (text + speaker) |
| **Model Size** | ~612M parameters |

## Installation

```bash
uv sync
```

## Usage

### Inference

```python
from drift_tts.inference import DriftTTSInference

tts = DriftTTSInference("path/to/checkpoint.pt")
waveform = tts.generate(
    text="Hello, this is Drift TTS speaking.",
    ref_audio="reference.wav",
    cfg_text=2.0,
    cfg_spk=2.0,
)
tts.save_audio(waveform, "output.wav")
```

CLI:
```bash
uv run python scripts/inference.py \
    --checkpoint path/to/checkpoint.pt \
    --text "Hello world" \
    --ref_audio reference.wav \
    --output output.wav
```

### Training

#### 1. Prepare Manifest

```bash
uv run python scripts/prepare_manifest.py \
    --audio_root /data/emilia/audio \
    --output /data/emilia/train.jsonl
```

Output JSONL format:
```json
{"audio_path": "spk001/0001.wav", "text": "Hello world", "speaker_id": "spk001"}
```

#### 2. Cache Latents (offline preprocessing)

```bash
uv run python scripts/cache_latents.py \
    --manifest /data/emilia/train.jsonl \
    --audio_root /data/emilia/audio \
    --output /data/emilia/cache
```

#### 3. Train

Single GPU:
```bash
uv run python scripts/train.py --config configs/train_500m.yaml --workdir runs/exp1
```

Multi-GPU (DDP):
```bash
uv run torchrun --nproc_per_node=8 scripts/train.py \
    --config configs/train_500m.yaml --workdir runs/exp1
```

Debug (small model):
```bash
uv run python scripts/train.py --config configs/train_debug.yaml --workdir runs/debug
```

#### 4. Evaluate

```bash
uv run python scripts/evaluate.py \
    --checkpoint runs/exp1/checkpoints/step_00500000.pt \
    --test_manifest test.jsonl \
    --output_dir eval_outputs
```

## Project Structure

```
drift_tts/
├── models/
│   ├── generator.py          # AudioDiTGen — main model
│   ├── dit_block.py          # AudioDiTBlock (self-attn + cross-attn + FFN + AdaLN)
│   ├── attention.py          # 1D Self-Attention + Cross-Attention + RoPE
│   ├── text_encoder.py       # UTF-8 byte Text Transformer (Echo TTS style)
│   ├── speaker_encoder.py    # Dual-path speaker encoder
│   └── modules.py            # RMSNorm, SwiGLU, TimestepEmbedder
├── loss/
│   ├── drift_loss.py         # Drift loss (PyTorch port)
│   └── memory_bank.py        # Speaker-cluster memory bank
├── features/
│   └── wavlm_extractor.py   # WavLM multi-scale feature extraction
├── codec/
│   └── dacvae.py             # DACVAE encode/decode wrapper
├── data/
│   ├── dataset.py            # Emilia / cached latent dataset loaders
│   ├── latent_cache.py       # Offline latent pre-encoding
│   ├── text_processing.py    # UTF-8 byte tokenizer
│   └── audio_processing.py   # Resample, chunk, pad
├── training/
│   ├── trainer.py            # Training loop with DDP multi-GPU
│   ├── train_state.py        # EMA model
│   └── scheduler.py          # LR warmup + schedule
├── inference/
│   └── generate.py           # One-step TTS inference + dual CFG
└── utils/
    ├── checkpoint.py         # Save/restore checkpoints
    ├── logging.py            # W&B + console logging
    └── misc.py               # Config loading, EasyDict

scripts/
├── train.py                  # Training entrypoint
├── inference.py              # Inference CLI
├── cache_latents.py          # Data preprocessing
└── evaluate.py               # Evaluation (UTMOS, WER, similarity, RTF)

configs/
├── train_500m.yaml           # Full 612M model config
└── train_debug.yaml          # Small model for debugging
```

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Total steps | 500,000 |
| Batch size | 32 per GPU |
| Learning rate | 2e-4 (constant + warmup) |
| EMA decay | 0.999 |
| Chunk duration | 10s (862 DACVAE frames) |
| R_list | [0.02, 0.05, 0.2] |
| CFG range | [1.0, 3.0] |
| Gradient clipping | 2.0 |

## References

- **Drift**: [Generative Modeling via Drifting](https://arxiv.org/abs/2602.04770) — original paradigm
- **Echo TTS**: [Echo](https://jordandarefsky.com/blog/2025/echo/) — UTF-8 byte text encoding reference
- **DACVAE**: [facebook/dacvae-watermarked](https://github.com/facebookresearch/dacvae) — audio codec
- **WavLM**: [WavLM-Large](https://huggingface.co/microsoft/wavlm-large) — feature extractor

## License

MIT
