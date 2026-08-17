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
| ASR      | `whisper-rs` in-process | features `cuda` + `vulkan` both enabled; runtime device pick via `WhisperContextParameters::gpu_device` (verified: whisper.cpp enumerates all registered GPU backends, whisper-rs passes the field through) |
| Paste    | clipboard + simulated Ctrl+V (`wl-copy`/`wl-paste` + `ydotool`), restore previous clipboard | works everywhere Ctrl+V works |
| History  | SQLite (`rusqlite`, bundled), text + language + timestamp, kept forever | audio not stored |

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
- Manual path override in settings.

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
