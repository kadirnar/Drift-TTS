"""Pre-encode Emilia dataset to DACVAE latents."""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Cache DACVAE latents")
    parser.add_argument("--manifest", type=str, required=True, help="Path to train.jsonl manifest.")
    parser.add_argument("--audio_root", type=str, required=True, help="Root directory for audio files.")
    parser.add_argument("--output", type=str, required=True, help="Output cache directory.")
    parser.add_argument("--dacvae_path", type=str, default="facebook/dacvae-watermarked")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_entries", type=int, default=None)
    args = parser.parse_args()

    from drift_tts.data.latent_cache import cache_latents

    cache_latents(
        manifest_path=args.manifest,
        audio_root=args.audio_root,
        output_root=args.output,
        dacvae_model_path=args.dacvae_path,
        device=args.device,
        max_entries=args.max_entries,
    )


if __name__ == "__main__":
    main()
