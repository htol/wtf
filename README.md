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

## NVIDIA note

WebKitGTK's DMA-BUF renderer crashes with `Error 71 (Protocol error)` on the
NVIDIA proprietary driver under Wayland, so the systemd unit sets
`WEBKIT_DISABLE_DMABUF_RENDERER=1`. If launched manually, export it first.
