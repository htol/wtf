# wtf — design record

Local dictation app: press hotkey, speak, press again, transcript is pasted into
the focused application. Personal tool for one machine: Linux, KDE Plasma 6,
Wayland. Stack: Rust, Tauri 2, whisper.cpp (whisper-rs), Svelte 5 + Vite, Nord.

## Product

- System-wide dictation (superwhisper-style). Core loop:
  hotkey -> record -> transcribe -> paste immediately (no preview step).
- Settings + history UI in one window with tabs.
- File transcription / live captions: out of scope for v1.
- LLM post-processing (llama.cpp): not in MVP. Architectural seam only
  (transcript pipeline step between whisper and paste); wire `llama-cpp-2`
  later without restructuring.

## Pipeline

| Stage    | Choice | Notes |
|----------|--------|-------|
| Hotkey   | xdg-desktop-portal GlobalShortcuts via `ashpd` (feature `global_shortcuts`) | Plasma 6 native binding dialog; press-to-toggle |
| Capture  | `cpal` in-process, default input device | resample to 16 kHz mono f32; risk: pipewire-alsa default routing — smoke test early |
| ASR      | whisper (`whisper-rs`) + GigaAM and Qwen3-ASR (`ort`/ONNX Runtime, see "Engines") | whisper: features `cuda` + `vulkan` both enabled; runtime device pick via `WhisperContextParameters::gpu_device` (verified: whisper.cpp enumerates all registered GPU backends, whisper-rs passes the field through). GigaAM, Qwen3-ASR: CPU EP only |
| Paste    | clipboard + simulated Ctrl+V (`wl-copy`/`wl-paste` + `ydotool`), restore previous clipboard | works everywhere Ctrl+V works |
| History  | SQLite (`rusqlite`, bundled), text + language + timestamp, kept forever | audio not stored |

## Engines

Three ASR engines behind one seam (`asr::Transcriber`):

- Routing: the `engine` setting (Whisper / GigaAM / Qwen3-ASR), separate
  from the language. GigaAM is Russian only and ignores the language;
  without its model -> whisper + a one-time-per-session desktop
  notification suggesting the download. Qwen3-ASR without a model fails
  and opens Settings, like whisper. (Before Qwen3-ASR the language
  selector doubled as the engine switch with two Russian rows; settings
  files from then are migrated on load.)
- GigaAM model: `gigaam-v3-e2e-ctc` from HF `istupakov/gigaam-v3-onnx`,
  revision pinned in code (`models::GIGAAM_HF_BASE`). Catalog: fp32
  (recommended — measured faster than int8 on CPU) and int8 (compact).
  Stored under `~/.local/share/wtf/models/gigaam/`.
- GigaAM internals (`gigaam.rs`): vendored log-mel preprocessor graph (from
  the onnx-asr wheel, MIT) run as-is via `ort`; CTC greedy decode
  (argmax per 40 ms frame, drop blank, collapse repeats; verified
  byte-identical to `onnx_asr.recognize` on the official sample).
- Long recordings: split with Silero VAD (`vad.rs`; vendored, MIT) into
  speech islands. GigaAM (segment limit ~25 s): each island separately,
  capped at 20 s. Qwen3-ASR: islands joined into pieces of up to 30 s, the
  language detected on the first piece forced on the rest. Whisper path
  unchanged (no length limit).
- Qwen3-ASR model: int4 ONNX exports of the 0.6B and 1.7B models from HF
  `andrewleech/qwen3-asr-{0.6b,1.7b}-onnx`, revisions pinned in code
  (`models::QWEN_CHOICES`). One directory per entry under
  `~/.local/share/wtf/models/qwen/` (encoder, two decoder graphs sharing
  one weights file, fp16 embedding table, tokenizer, config).
- Qwen3-ASR internals (`qwen.rs`): Whisper-style log-mel (128 bins)
  computed in Rust (`rustfft`); encoder -> decoder prefill -> greedy
  token loop with KV cache. The model answers
  `language <Name><asr_text><transcript>`; a chosen language pre-fills
  that prefix. Verified token-identical to a librosa + onnxruntime
  pipeline on the official GigaAM sample. Tokenizer: decode only, straight
  from `tokenizer.json` (no `tokenizers` crate).
- Qwen3-ASR limits, measured on the official GigaAM samples: with 0.6B
  Auto can fail on Russian (the model paraphrased the 11 s sample in
  English with no language tag — reported as an error, the fix is picking
  the language; 1.7B detected Russian on the same sample). Decoding cost
  grows faster than length (0.6B: 71 s in one piece -> 41 s, in 30 s
  pieces -> 20 s; 1.7B in pieces -> 31 s).
- All engines stay cached in app state once loaded (no eviction on
  engine switch); `unload_model` drops everything.
- Prompts: GigaAM and Qwen3-ASR ignore `initial_prompt`; the Prompts tab
  carries a static banner saying prompts apply to whisper only.
- CPU only for Qwen3-ASR as well (same `ort` build).
- CPU only for GigaAM — final: GPU rejected (CPU latency already sits
  below the dictation perceptibility threshold; a GPU session would only
  add resident VRAM and an `ort` `rocm`/`webgpu` build). `ort` is pinned
  `=2.0.0-rc.13` and fetches prebuilt libonnxruntime at build time
  (`download-binaries`).

## UX

- Recording indicator: small floating always-on-top window, centered
  horizontally, initial position 20% from bottom, draggable, remembers
  position; shows signal level, language, status.
- Languages: multilingual models, manual selection + `auto` mode
  (whisper language detection). Switch via tray menu and a global
  "cycle languages" hotkey through the same portal.
- Errors / cancellation: Esc cancels a stuck transcription; failures raise a
  desktop notification (portal) + red indicator flash; history stores only
  successful transcriptions.
- Speech-to-English translation (`task=translate`): not in MVP.

## Models

- First run: picker (tiny ... large-v3-turbo), download to
  `~/.local/share/wtf/models`. Default recommendation: `large-v3-turbo` q5_0.
- GigaAM and Qwen3-ASR models: separate catalogs,
  `~/.local/share/wtf/models/gigaam/` and `.../qwen/` (see "Engines").
- Manual path override in settings (whisper only).
- Update check (Settings button): sha256 of each installed model vs the HF
  tree API (`lfs.oid`) — whisper against `main`, GigaAM and Qwen3-ASR against
  their pinned revisions; a mismatch means "re-download".

## App identity

- Working name `wtf`. `src-tauri/src/app_id.rs` is the single source of truth
  for the app id and data/config paths, so a later rename is one change +
  a one-time data-directory migration.
- Tauri identifier: `local.wtf.app`.

## Launch / install

- systemd user unit (`assets/wtf.service`), installed by `make install` to
  `~/.config/systemd/user/`, enabled by `make enable`.
- Makefile-driven (no ad-hoc scripts). No bundling (no AppImage/deb);
  binary goes to `~/.local/bin`.

## Known risks / to verify early

1. Dual-backend link (cuda + vulkan in one binary) compiles on paper; the
   `make smoke` build confirms or refutes it. Fallback: separate feature
   profiles per backend.
2. Always-on-top overlay window on Wayland via GTK — prototype in week one.
3. cpal default-device routing under PipeWire — smoke test in week one.
4. `ydotool` needs `ydotoold` running; clipboard restore can race with
   clipboard managers (mitigated by a short delay, see `inject.rs`).
