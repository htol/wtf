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
| ASR      | whisper (`whisper-rs`) + GigaAM (`ort`/ONNX Runtime) + Qwen3-ASR (`llama-server` child process), see "Engines" | whisper: features `cuda` + `vulkan` both enabled; runtime device pick via `WhisperContextParameters::gpu_device` (verified: whisper.cpp enumerates all registered GPU backends, whisper-rs passes the field through). GigaAM: CPU EP only. Qwen3-ASR: Vulkan or CPU, own settings (`qwen_use_gpu`, `qwen_gpu_device` as a llama.cpp device name — its order need not match whisper's indices) |
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
- Long recordings (GigaAM path only; its segment limit is ~25 s): split
  with Silero VAD (`vad.rs`; vendored, MIT) into speech islands, each
  capped at 20 s. Whisper and Qwen3-ASR take a recording whole.
- Qwen3-ASR model: Q8_0 GGUF of the 0.6B and 1.7B models from HF
  `ggml-org/Qwen3-ASR-{0.6B,1.7B}-GGUF`, revisions pinned in code
  (`models::QWEN_CHOICES`). One directory per entry under
  `~/.local/share/wtf/models/qwen/` (language model + audio encoder
  "mmproj").
- Qwen3-ASR runtime: llama.cpp cannot be linked in next to whisper.cpp
  (each carries its own ggml), so the model runs in a `llama-server` child
  process. The app downloads the pinned release's prebuilt Vulkan build
  (`models::LLAMA_BUILD`, sha256-checked, unpacked with `tar`) into
  `~/.local/share/wtf/llama/` together with the first model.
- Qwen3-ASR internals (`qwen.rs`): the server starts on the first
  dictation — loopback port, per-process API key, one slot, all layers on
  the chosen GPU or none (`--device none`), explicit `--ctx-size`
  (`qwen_context`, default 8192; without it llama.cpp starts from the
  model's 65536 tokens; audio takes ~13 tokens/s) — and stops when the
  engine is dropped (`unload_model`, model or Qwen setting change); the
  systemd unit's cgroup covers an app crash. A recording is
  one chat request with a base64 WAV. The model answers
  `language <Name><asr_text><transcript>`; a chosen language pre-fills
  that prefix as the start of the assistant turn.
- Qwen3-ASR measurements (RX 9070 XT, official GigaAM samples): 11 s of
  Russian in 0.33 s (0.6B) / 0.57 s (1.7B), 71 s in 1.6 s / 2.7 s, model
  load ~1 s. On an i7-8565U with UHD 620 the CPU beats the iGPU: 12.5 s of
  speech end to end incl. load, 0.6B 7.3 s vs 11.0 s, 1.7B 17.9 s vs
  31.4 s. Digital silence makes it hallucinate (Chinese text); the
  pipeline's silence guard runs before the engine. An earlier int4 ONNX
  build on the CPU (`ort`) was 8-13x slower and less accurate on Russian,
  and was replaced.
- All engines stay cached in app state once loaded (no eviction on
  engine switch); `unload_model` drops everything.
- Prompts: GigaAM and Qwen3-ASR ignore `initial_prompt`; the Prompts tab
  carries a static banner saying prompts apply to whisper only.
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
- Tauri identifier: `htol.wtf`.

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
