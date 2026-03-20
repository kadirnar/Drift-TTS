"""TTS dataset için JSONL manifest oluşturur.

Desteklenen formatlar:
  1. Emilia: her ses dosyasının yanında aynı isimli .txt dosyası var
  2. LJSpeech tarzı: metadata.csv (dosya_adı|transkript) formatı
  3. LibriTTS tarzı: .normalized.txt dosyaları

Çıktı formatı (her satır bir JSON):
  {"audio_path": "spk001/0001.wav", "text": "Hello world", "speaker_id": "spk001"}

Kullanım:
  uv run python scripts/prepare_manifest.py \
      --audio_root /data/emilia/audio \
      --output /data/emilia/train.jsonl
"""

import argparse
import json
from pathlib import Path

import torchaudio
from tqdm import tqdm


def get_duration(audio_path: str) -> float:
    info = torchaudio.info(audio_path)
    return info.num_frames / info.sample_rate


def find_text_for_audio(audio_path: Path) -> str:
    """Ses dosyasına karşılık gelen metni bulmaya çalışır."""
    for suffix in (".txt", ".normalized.txt", ".lab"):
        txt = audio_path.with_suffix(suffix)
        if txt.exists():
            return txt.read_text(encoding="utf-8").strip()
    return ""


def infer_speaker_id(audio_path: Path, audio_root: Path) -> str:
    """Klasör yapısından speaker ID çıkarır (ilk alt klasör adı)."""
    rel = audio_path.relative_to(audio_root)
    parts = rel.parts
    return parts[0] if len(parts) > 1 else "default"


def build_from_directory(
    audio_root: Path,
    extensions: tuple = (".wav", ".flac", ".mp3", ".ogg"),
    min_duration: float = 0.5,
    max_duration: float = 30.0,
) -> list:
    """audio_root altındaki tüm ses dosyalarını tarar."""
    entries = []
    audio_files = []
    for ext in extensions:
        audio_files.extend(audio_root.rglob(f"*{ext}"))
    audio_files.sort()

    for audio_path in tqdm(audio_files, desc="Scanning audio files"):
        try:
            dur = get_duration(str(audio_path))
        except Exception:
            continue

        if dur < min_duration or dur > max_duration:
            continue

        text = find_text_for_audio(audio_path)
        if not text:
            continue

        entries.append({
            "audio_path": str(audio_path.relative_to(audio_root)),
            "text": text,
            "speaker_id": infer_speaker_id(audio_path, audio_root),
        })

    return entries


def build_from_metadata_csv(
    audio_root: Path,
    metadata_path: Path,
    delimiter: str = "|",
    min_duration: float = 0.5,
    max_duration: float = 30.0,
) -> list:
    """LJSpeech tarzı metadata.csv dosyasından manifest oluşturur."""
    entries = []
    with open(metadata_path, encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading metadata"):
            line = line.strip()
            if not line:
                continue
            parts = line.split(delimiter)
            if len(parts) < 2:
                continue

            file_id = parts[0].strip()
            text = parts[-1].strip()

            audio_path = None
            for ext in (".wav", ".flac", ".mp3"):
                for subdir in ("", "wavs"):
                    candidate = audio_root / subdir / f"{file_id}{ext}" if subdir else audio_root / f"{file_id}{ext}"
                    if candidate.exists():
                        audio_path = candidate
                        break
                if audio_path:
                    break

            if audio_path is None:
                continue

            try:
                dur = get_duration(str(audio_path))
            except Exception:
                continue

            if dur < min_duration or dur > max_duration:
                continue

            entries.append({
                "audio_path": str(audio_path.relative_to(audio_root)),
                "text": text,
                "speaker_id": infer_speaker_id(audio_path, audio_root),
            })

    return entries


def main():
    parser = argparse.ArgumentParser(description="TTS dataset JSONL manifest olusturucu")
    parser.add_argument("--audio_root", type=str, required=True, help="Ses dosyalarinin kok dizini.")
    parser.add_argument("--output", type=str, required=True, help="Cikti JSONL yolu.")
    parser.add_argument("--format", type=str, default="auto", choices=["auto", "directory", "metadata_csv"])
    parser.add_argument("--metadata_csv", type=str, default=None, help="metadata.csv yolu.")
    parser.add_argument("--delimiter", type=str, default="|")
    parser.add_argument("--min_duration", type=float, default=0.5)
    parser.add_argument("--max_duration", type=float, default=30.0)
    args = parser.parse_args()

    audio_root = Path(args.audio_root).resolve()

    # Format algılama
    fmt = args.format
    if fmt == "auto":
        csv_path = args.metadata_csv
        if csv_path is None:
            for name in ("metadata.csv", "metadata.txt", "transcripts.csv"):
                p = audio_root / name
                if p.exists():
                    csv_path = str(p)
                    break
        fmt = "metadata_csv" if csv_path and Path(csv_path).exists() else "directory"

    if fmt == "metadata_csv":
        csv_path = Path(args.metadata_csv or audio_root / "metadata.csv")
        print(f"Format: metadata_csv ({csv_path})")
        entries = build_from_metadata_csv(audio_root, csv_path, args.delimiter, args.min_duration, args.max_duration)
    else:
        print(f"Format: directory scan ({audio_root})")
        entries = build_from_directory(audio_root, min_duration=args.min_duration, max_duration=args.max_duration)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    total_dur = 0.0
    for e in entries:
        try:
            total_dur += get_duration(str(audio_root / e["audio_path"]))
        except Exception:
            pass
    speakers = set(e["speaker_id"] for e in entries)
    print(f"\n{output_path}")
    print(f"  Entries:  {len(entries):,}")
    print(f"  Speakers: {len(speakers):,}")
    print(f"  Duration: {total_dur/3600:.1f} hours")


if __name__ == "__main__":
    main()
