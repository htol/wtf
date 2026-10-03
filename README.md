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
(0.6B or 1.7B, int4 ONNX, CPU) is multilingual with language detection; its
model files live in `~/.local/share/wtf/models/qwen/`. The first build
downloads a prebuilt libonnxruntime (cargo feature `download-binaries` of
`ort`).

## Runtime dependencies

- `wl-copy`, `wl-paste` (clipboard injection)
- `ydotool` + running `ydotoold` (simulated Ctrl+V)
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

## Wayland notes

- The recording overlay needs a KWin window rule: xdg-shell has no
  keep-above, so `alwaysOnTop` from the app is ignored. Add to
  `~/.config/kwinrulesrc` (and `qdbus org.kde.KWin /KWin reconfigure` or
  re-login — KWin applies rules when a window is created, so restart the
  service afterwards):

  ```ini
  [General]
  rules=1

  [1]
  Description=wtf overlay: keep above, skip taskbar and switcher, remember position
  title=wtf-overlay
  titlematch=1
  above=true
  aboverule=3
  skiptaskbar=true
  skiptaskbarrule=3
  skipswitcher=true
  skipswitcherrule=3
  positionrule=2
  types=1
  ```

  Every property needs its paired `*rule=3` (Force) field; without them KWin
  silently ignores the rule. `positionrule=2` (Remember) makes KWin store the
  overlay position — client-side positioning is impossible on Wayland.

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

The overlay rules replace the KWin rule above (0.56 syntax; every field is
`<effect> <value>` or `match:<prop> <value>`). `pin` covers keep-above.
`no_initial_focus` matters because the overlay is mapped on every recording
(GTK3 cannot resize a mapped Wayland toplevel — `gtk_window_resize` is
silently ignored after map — so the app shows/hides it instead of resizing)
and the map must not steal focus from the dictation target.

```ini
windowrule = float true, match:title ^(wtf-overlay)$
windowrule = pin true, match:title ^(wtf-overlay)$
windowrule = decorate false, match:title ^(wtf-overlay)$
windowrule = no_initial_focus true, match:title ^(wtf-overlay)$
```

## NVIDIA note

WebKitGTK's DMA-BUF renderer crashes with `Error 71 (Protocol error)` on the
NVIDIA proprietary driver under Wayland, so the systemd unit sets
`WEBKIT_DISABLE_DMABUF_RENDERER=1`. If launched manually, export it first.
