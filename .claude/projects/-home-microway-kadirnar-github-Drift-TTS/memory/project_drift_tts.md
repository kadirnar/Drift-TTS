---
name: drift_tts_implementation
description: Drift-TTS project - one-step TTS using drift paradigm, ported from JAX to PyTorch with audio pipeline
type: project
---

Drift-TTS: PyTorch port of the Drift generative paradigm (arXiv:2602.04770) applied to TTS.

**Why:** First application of drift model to TTS - enables one-step high-quality speech generation.

**How to apply:**
- Source JAX code is in `drifting/` directory
- PyTorch implementation is in `drift_tts/` package
- Uses `uv` for dependency management (NOT pip/conda)
- DACVAE installed from `git+https://github.com/facebookresearch/dacvae` (not PyPI `dac`)
- Full model is 612.2M params (hidden_size=1024, depth=22, num_heads=16)
- All tests pass: drift loss, memory bank, all model components, forward pass shapes
