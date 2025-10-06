#!/usr/bin/env python3
"""
Audio transcription script using OpenAI Whisper with MLX framework
"""

import mlx_whisper
import sys
import argparse
import time


def transcribe_audio(audio_file, model_size="base", language=None):
    """
    Transcribe an audio file using Whisper with MLX

    Args:
        audio_file: Path to the audio file
        model_size: Whisper model size (tiny, base, small, medium, large)
        language: Optional language code (e.g., 'en', 'es', 'fr')

    Returns:
        Transcription result
    """
    print(f"Transcribing with MLX Whisper model: {model_size}")

    # MLX Whisper uses transcribe directly with model parameter
    options = {
        "path_or_hf_repo": f"mlx-community/whisper-turbo",
        "verbose": False
    }

    if language:
        options["language"] = language

    result = mlx_whisper.transcribe(audio_file, **options)

    return result


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio files using OpenAI Whisper with MLX")
    parser.add_argument("audio_file", help="Path to the audio file")
    parser.add_argument("-m", "--model", default="base",
                        choices=["tiny", "base", "small", "medium", "large","turbo"],
                        help="Whisper model size (default: base)")
    parser.add_argument("-l", "--language", help="Language code (e.g., en, es, fr)")
    parser.add_argument("-o", "--output", help="Output file for transcription (optional)")

    args = parser.parse_args()

    try:
        start_time = time.time()
        result = transcribe_audio(args.audio_file, args.model, args.language)

        print("\n" + "="*50)
        print(f"TRANSCRIPTION ({time.time() - start_time:.2f}s):")
        print("="*50)
        print(result["text"])

        if args.output:
            with open(args.output, 'w') as f:
                f.write(result["text"])
            print(f"\nTranscription saved to: {args.output}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
