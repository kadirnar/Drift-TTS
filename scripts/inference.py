"""Inference CLI for Drift-TTS."""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Drift-TTS Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize.")
    parser.add_argument("--ref_audio", type=str, required=True, help="Path to reference audio.")
    parser.add_argument("--output", type=str, default="output.wav", help="Output WAV path.")
    parser.add_argument("--cfg_text", type=float, default=2.0, help="Text CFG scale.")
    parser.add_argument("--cfg_spk", type=float, default=2.0, help="Speaker CFG scale.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Noise temperature.")
    parser.add_argument("--no_trim", action="store_true", help="Disable trailing silence trimming.")
    parser.add_argument("--device", type=str, default="cuda", help="Device.")
    args = parser.parse_args()

    from drift_tts.inference.generate import DriftTTSInference

    tts = DriftTTSInference(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    waveform = tts.generate(
        text=args.text,
        ref_audio=args.ref_audio,
        cfg_text=args.cfg_text,
        cfg_spk=args.cfg_spk,
        temperature=args.temperature,
        trim_silence=not args.no_trim,
    )

    tts.save_audio(waveform, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
