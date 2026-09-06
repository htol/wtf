# PROTOTYPE — GigaAM feasibility spike (throwaway)

Answers: can this machine run GigaAM v3 (Russian ASR) locally, how fast, and
does the e2e variant give dictation-ready punctuated text? If yes, the plan is
a second in-process Rust engine next to whisper (`ort` + ONNX), see the chat
discussion on `main`.

Run (downloads the official GigaAM sample wav, model comes from
`istupakov/gigaam-v3-onnx` via the HF cache on first run):

```sh
make spike-gigaam                    # e2e ctc, int8 (default)
make spike-gigaam MODEL=gigaam-v3-e2e-rnnt
```

## Verdict

**YES — GigaAM v3 ONNX runs locally on plain CPU** (onnxruntime CPU EP, no GPU,
no CUDA/ROCm needed). Measured on the official 11.3s sample (`example.wav`):

| model         | quant | recognize (cold/warm) | RTFx   |
|---------------|-------|-----------------------|--------|
| e2e-ctc       | int8  | 0.34s                 | 32.8   |
| e2e-rnnt      | int8  | 0.51s / 0.44s         | 22.2   |
| e2e-ctc       | fp32  | 0.32s / 0.26s         | 35.2/43.9 |

- e2e variants return punctuated text — dictation-ready. rnnt capitalizes
  line starts; ctc keeps prose lowercase after the first word.
- fp32 beats int8 on this CPU (QLinearConv overhead) — and the CTC decode is
  a plain argmax, so for the Rust port `gigaam-v3-e2e-ctc` fp32 via `ort` is
  both the fastest CPU path and the simplest to implement. RNNT is ~1.5x
  slower here and needs the decoder/joint loop.
- First `load_model` includes the HF download (~0.9GB fp32 / ~0.3GB int8);
  warm load is disk-bound (~seconds).
- Next check before porting: real mic dictations vs whisper large-v3-turbo
  (quality is the actual reason to add the engine; latency already passes).
