#!/usr/bin/env python3
"""
Audio transcription script using OpenAI Whisper
"""

import whisper
import sys
import argparse
import time


def transcribe_audio(audio_file, model_size="base", language=None):
    """
    Transcribe an audio file using Whisper

    Args:
        audio_file: Path to the audio file
        model_size: Whisper model size (tiny, base, small, medium, large)
        language: Optional language code (e.g., 'en', 'es', 'fr')

    Returns:
        Transcription result
    """
    print(f"Loading Whisper model: {model_size}")
    model = whisper.load_model(model_size)

    print(f"Transcribing: {audio_file}")
    result = model.transcribe(audio_file, language=language)

    return result


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio files using OpenAI Whisper")
    parser.add_argument("audio_file", help="Path to the audio file")
    parser.add_argument("-m", "--model", default="base",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: base)")
    parser.add_argument("-l", "--language", help="Language code (e.g., en, es, fr)")
    parser.add_argument("-o", "--output", help="Output file for transcription (optional)")

    args = parser.parse_args()

    try:
        start_time = time.time()
        result = transcribe_audio(args.audio_file, args.model, args.language)

        print("\n" + "="*50)
        print(f"TRANSCRIPTION ({time.time() - start_time}):")
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
