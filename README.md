# wtf - local voice-to-text dictation (Tauri 2 + whisper.cpp + GigaAM)

## Decisions

See `DESIGN.md` for the full decision record.

## Development

```sh
make dev        # tauri dev (hot reload, CPU whisper build)
make build      # release build into src-tauri/target/release/wtf
make smoke      # debug build with asr-cuda+asr-vulkan (validates dual-backend link)
make install    # install binary + desktop file + systemd user unit (~/.local/bin,
                 # ~/.local/share/applications, ~/.config/systemd/user)
make enable     # systemctl --user enable --now app-wtf.service
make check      # cargo check (default features)
make clean      # clean cargo + vite artifacts
```

## Engines

The engine is picked in Settings, General. Whisper handles auto-detect and
all languages. GigaAM v3 (`ort`/ONNX Runtime, CPU) is Russian only; its model
files live in `~/.local/share/wtf/models/gigaam/`, and the mel preprocessor
and Silero VAD graphs are vendored in `src-tauri/assets/` (MIT). Qwen3-ASR
(0.6B or 1.7B, Q8 GGUF) is multilingual with language detection and runs on
a GPU or the CPU in a `llama-server` child process; its model files live in
`~/.local/share/wtf/models/qwen/`, and the app downloads a pinned llama.cpp
Vulkan build into `~/.local/share/wtf/llama/` with the first model. The
first build downloads a prebuilt libonnxruntime (cargo feature
`download-binaries` of `ort`).

## Runtime dependencies

- `wl-copy`, `wl-paste` (clipboard injection)
- `ydotool` + running `ydotoold` (simulated Ctrl+V)
- `tar` and a Vulkan driver (Qwen3-ASR engine: unpacking and running the
  llama.cpp build)
- xdg-desktop-portal (GlobalShortcuts, notifications) — stock KDE Plasma 6
  (Hyprland: see "Wayland notes, Hyprland"; `xdg-desktop-portal-hyprland`).
  The systemd unit is named `app-wtf.service` and `wtf.desktop` is installed:
  the portal derives the app id for unsandboxed apps from the `app-*`
  user-unit name plus a matching desktop file.
- Plasma session restore relaunches the app at login even though the
  systemd unit already started it (restore's dedup only covers XDG
  autostart and ksmserver clients). Exclude the app in
  `~/.config/ksmserverrc` — and the binary refuses to run as a second
  instance regardless (tauri-plugin-single-instance):

  ```ini
  [General]
  excludeApps=wtf
  ```

## macOS

`make build` produces `src-tauri/target/release/bundle/macos/wtf.app`
(whisper on the GPU via Metal), `make install` copies it to
`~/Applications`, `make enable` starts it now and at every login (a
LaunchAgent in `~/Library/LaunchAgents`). `make smoke` is Linux only.
Building needs `cmake` (whisper.cpp): `brew install cmake`.

- Shortcuts are set in Settings, Shortcuts (default: Option+` toggles
  recording); there is no system binding dialog.
- Pasting sends Cmd+V and needs the Accessibility permission (System
  Settings, Privacy & Security); the microphone prompt appears on the first
  recording. The app asks for both.
- macOS ties these grants to the code signature. By default the bundle is
  ad-hoc signed and loses them on every rebuild, leaving a stale entry in
  the Accessibility list (`tccutil reset Accessibility htol.wtf`
  clears it). To keep them, the first `make build` creates a self-signed
  `wtf-dev` certificate in the login keychain and signs with it from then
  on; answer Always Allow when codesign asks for the key. Creating it
  resets the grants once. `SIGN_IDENTITY=<name>` picks another identity.
- Builds before the rename to `htol.wtf` used the identifier
  `local.wtf.app`; `make purge-old-id` removes its grants, webview data and
  login agent (run `make enable` again afterwards).
- `make dev` runs unbundled: the permissions then belong to the terminal
  that started it. Unlike on Linux, it runs whisper on the GPU (Metal).
- Qwen3-ASR downloads the pinned llama.cpp Metal build instead of the
  Vulkan one.

## Wayland notes

- The recording overlay is a wlr-layer-shell surface (gtk-layer-shell;
  KWin since Plasma 5.20, Hyprland, Sway). xdg-shell has no keep-above, no
  client-side position and no way to refuse focus, so a plain window would
  need compositor rules and every map (the overlay is shown on each
  recording — GTK3 cannot resize a mapped Wayland toplevel) would activate
  it and take focus from the dictation target. As a layer surface it sits
  in the overlay layer, never gets keyboard focus and places itself from
  the saved position; drag the pill to move it. No KWin or Hyprland window
  rules are needed. Runtime dependency: the `gtk-layer-shell` library
  (Arch `gtk-layer-shell`, Debian `libgtk-layer-shell0`). On X11 the
  overlay stays a plain window.

### Hyprland

The same `app-*` unit story applies: launch with `make enable`, never from a
terminal. A process outside an `app-*` unit has no portal app id, so
GlobalShortcuts registration fails ("An app id is required"), the hotkey hub
is never put into app state, and the Settings rebind button then hangs
silently (its command panics on the missing state).

There is no binding dialog — that is a Plasma feature. With
`xdg-desktop-portal-hyprland`, keys are bound in hyprland.conf via the
`global` dispatcher; `hyprctl globalshortcuts` lists the registered ids
(`wtf:record`, `wtf:cycle-language`):

```ini
bind = ALT, grave, global, wtf:record
```

## NVIDIA note

WebKitGTK's DMA-BUF renderer crashes with `Error 71 (Protocol error)` on the
NVIDIA proprietary driver under Wayland, so the systemd unit sets
`WEBKIT_DISABLE_DMABUF_RENDERER=1`. If launched manually, export it first.
