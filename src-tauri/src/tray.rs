//! System tray: quick access to the settings window and language switching.
//!
//! The tray icon mirrors the recording state: blue circle when idle, red
//! while recording (DESIGN.md "UX"; the floating overlay comes later).

use tauri::menu::{Menu, MenuItem};
use tauri::tray::TrayIconBuilder;
use tauri::{image::Image, Manager};

const IDLE_ICON: &[u8] = include_bytes!("../icons/tray-idle.png");
const RECORDING_ICON: &[u8] = include_bytes!("../icons/tray-recording.png");

pub fn init(app: &tauri::App) -> tauri::Result<()> {
	let open = MenuItem::with_id(app, "open", "Open wtf", true, None::<&str>)?;
	let quit = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
	let menu = Menu::with_items(app, &[&open, &quit])?;

	let mut builder = TrayIconBuilder::with_id("main-tray")
		.tooltip("wtf")
		.menu(&menu)
		.on_menu_event(|app, event| match event.id.as_ref() {
			"open" => {
				if let Some(window) = app.get_webview_window("main") {
					let _ = window.show();
					let _ = window.set_focus();
				}
			}
			"quit" => app.exit(0),
			_ => {}
		});
	if let Ok(icon) = Image::from_bytes(IDLE_ICON) {
		builder = builder.icon(icon);
	} else if let Some(icon) = app.default_window_icon() {
		builder = builder.icon(icon.clone());
	}
	builder.build(app)?;
	Ok(())
}

/// Swaps the tray icon to the recording (red) or idle (blue) variant.
pub fn set_recording(app: &tauri::AppHandle, recording: bool) {
	let Some(tray) = app.tray_by_id("main-tray") else {
		return;
	};
	let bytes = if recording {
		RECORDING_ICON
	} else {
		IDLE_ICON
	};
	if let Ok(icon) = Image::from_bytes(bytes) {
		let _ = tray.set_icon(Some(icon));
	}
}
