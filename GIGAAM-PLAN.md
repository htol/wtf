# GigaAM engine — implementation plan (feat branch)

Working notes for this branch. Before merging, fold the durable decisions into
`DESIGN.md` and delete this file. Experiment record: branch `spike/gigaam`
(commit `4f91837`) — local run verdict, latency numbers, reproducible env.

## Decisions (settled, do not relitigate)

- Second ASR engine next to whisper inside the existing `asr.rs` seam. Invisible
  to the user except model catalog cards and one banner.
- Routing: language `ru` -> GigaAM when its model is downloaded; `auto` and all
  other languages -> whisper. `ru` without the model -> whisper + one-time
  notification suggesting the download. The language selector doubles as the
  engine switch — no extra knob.
- Model: `gigaam-v3-e2e-ctc` from HF `istupakov/gigaam-v3-onnx`, revision
  pinned in code. Two catalog entries: fp32 (recommended, ~0.9 GB) and int8
  (compact, ~0.3 GB). Stored under `~/.local/share/wtf/models/gigaam/`.
- CPU only. GPU is a separate follow-up (ort has `rocm`/`webgpu` features).
- Long recordings: split with Silero VAD before GigaAM (its segment limit is
  ~25 s); transcribe chunks, join. Whisper path unchanged (no length limit).
- Both engines stay cached in app state once loaded (no eviction on language
  switch). Existing `unload_model` drops everything.
- Prompts: GigaAM ignores `initial_prompt`. Static text banner in the Prompts
  section: prompts apply to Whisper only (auto mode / other languages); for
  Russian GigaAM is used and prompts do not apply. No greying, no hiding.
- History unchanged (`ru` as before). No acceptance gates; a parity smoke run
  is part of development hygiene, not a gate.

## Implementation constants (verified facts)

- `ort` crate `2.0.0-rc.13`: default features already include CPU provider and
  `download-binaries` (prebuilt onnxruntime fetched at build time). Dep of
  edition 2024 / rustc 1.88+ — fine as a dependency of our edition-2021 crate.
- Model graph: inputs `features` `[B, 64, T]` f32 + `feature_lengths` `[B]`
  (i64), output `log_probs` `[B, T', 257]`. Subsampling 4; output frame = 40 ms;
  `T' = (feature_lengths - 1) / 4 + 1`.
- Pinned HF revision: `322c3b29492673eb7d0b434bfa9dfb8653e34d02`
  (`istupakov/gigaam-v3-onnx`). Files: `v3_e2e_ctc.onnx`,
  `v3_e2e_ctc.int8.onnx`, `v3_e2e_ctc_vocab.txt`, `config.json`.
- Vocab: plain text, one `token id` per line, 257 entries; `▁` (U+2581) maps to
  space; blank is token `<blk>` with id 256; `<unk>` is id 0. CTC greedy
  decode: argmax per frame, drop blank, collapse consecutive repeats, map ids
  to tokens, join, collapse redundant spaces.
- Mel preprocessor: onnx-asr ships it as an ONNX graph inside the pip wheel
  (`onnx_asr/preprocessors/data/gigaam_v3.onnx`, MIT): inputs `waveforms`
  `[B, L]` f32 @16 kHz + `waveforms_lens` `[B]` i64 -> `features [B,64,T]` +
  `features_lens`. It is a plain torchaudio-style log-mel (Hann window 320 /
  hop 160, 64 mel bins, power spectrogram -> mel matmul -> clip -> log; no
  dither, no preemphasis, no per-feature normalization). Either run that graph
  via `ort` as-is (extract from the wheel, ship/download it) or port the ~5 ops
  to Rust (`rustfft`). Implementer's choice; running the graph is the
  zero-parity-risk default.
- Reference latencies on this machine (Ryzen CPU): fp32 0.26 s / int8 0.34 s
  per 11.3 s audio (warm). fp32 is the recommended card.
- VAD: silero VAD ONNX (~2 MB, MIT), run via the same `ort` session stack.

## Task breakdown (in order)

1. `models.rs`: GigaAM catalog entries (both quantizations, sizes, pinned HF
   URLs), download with progress, uninstall; storage dir as above.
2. `src-tauri/src/gigaam.rs` (new): ort sessions (preprocessor + model),
   vocab load, CTC greedy decode, `transcribe(samples: &[f32]) -> Result<String>`;
   wire into `asr.rs` as the second `Transcriber` variant.
3. `pipeline.rs`: routing per decisions; two-slot engine cache (whisper +
   gigaam) keyed by model path; one-time `ru`-without-model notification.
4. VAD chunking on the GigaAM path for recordings over ~20 s.
5. UI: GigaAM cards in the existing model picker; banner in `Prompts.svelte`.
6. Smoke check: recreate the spike venv (branch `spike/gigaam`,
   `uv run --project spike/gigaam python spike/gigaam/run.py`) and compare its
   output on `example.wav` with the Rust pipeline output — expect identical
   text. Dev hygiene only.
7. Wrap-up: README/DESIGN updates (engine table row, model section), remove
   this file, merge.
