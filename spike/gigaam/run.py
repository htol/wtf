#!/usr/bin/env python3
"""PROTOTYPE (throwaway): check that GigaAM v3 ONNX runs locally.

Answers three questions for the dictation app (wtf):
1. Does onnxruntime transcribe Russian speech on this machine's CPU?
2. How fast (RTFx = audio seconds per compute second)?
3. Does the e2e variant return punctuated text suitable for dictation?

Not production code: no error handling, no abstractions.

Usage: uv run run.py [--model NAME] [--fp32] [wav ...]
Default model: gigaam-v3-e2e-ctc (int8 ONNX from istupakov/gigaam-v3-onnx).
"""

import argparse
import time
import wave
from pathlib import Path

import onnx_asr

DEFAULT_WAV = Path(__file__).parent / "example.wav"


def duration_of(wav: str) -> float:
    with wave.open(wav) as w:
        return w.getnframes() / w.getframerate()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gigaam-v3-e2e-ctc")
    parser.add_argument("--fp32", action="store_true", help="use fp32 ONNX instead of int8")
    parser.add_argument("wavs", nargs="*", help="wav files (default: the GigaAM sample)")
    args = parser.parse_args()

    wavs = args.wavs or [str(DEFAULT_WAV)]

    start = time.perf_counter()
    quantization = None if args.fp32 else "int8"
    model = onnx_asr.load_model(args.model, quantization=quantization)
    print(f"# model={args.model} quantization={quantization} load={time.perf_counter() - start:.1f}s")

    for wav in wavs:
        duration = duration_of(wav)
        start = time.perf_counter()
        result = model.recognize(wav)
        elapsed = time.perf_counter() - start
        rtfx = duration / elapsed if elapsed else float("inf")
        print(f"# {wav}: {duration:.1f}s audio, recognize={elapsed:.2f}s, RTFx={rtfx:.1f}")
        print(result)


if __name__ == "__main__":
    main()
