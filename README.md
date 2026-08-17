# wtf - local voice-to-text dictation (Tauri 2 + whisper.cpp)

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

## Runtime dependencies

- `wl-copy`, `wl-paste` (clipboard injection)
- `ydotool` + running `ydotoold` (simulated Ctrl+V)
- xdg-desktop-portal (GlobalShortcuts, notifications) — stock KDE Plasma 6.
  The systemd unit is named `app-wtf.service` and `wtf.desktop` is installed:
  the portal derives the app id for unsandboxed apps from the `app-*`
  user-unit name plus a matching desktop file.

## NVIDIA note

WebKitGTK's DMA-BUF renderer crashes with `Error 71 (Protocol error)` on the
NVIDIA proprietary driver under Wayland, so the systemd unit sets
`WEBKIT_DISABLE_DMABUF_RENDERER=1`. If launched manually, export it first.
