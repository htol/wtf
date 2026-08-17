//! System tray: quick access to the settings window and language switching.
//!
//! TODO (DESIGN.md, "UX"): language submenu (auto + recent languages) and a
//! "cycle language" entry mirroring the global shortcut; recording state
//! reflected in the icon.

use tauri::menu::{Menu, MenuItem};
use tauri::tray::TrayIconBuilder;
use tauri::Manager;

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
	if let Some(icon) = app.default_window_icon() {
		builder = builder.icon(icon.clone());
	}
	builder.build(app)?;
	Ok(())
}
