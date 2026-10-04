//! Global recording hotkey on macOS via tauri-plugin-global-shortcut.
//!
//! macOS has no binding dialog like the Linux desktop portal, so the key
//! combinations live in settings (`record_shortcut`,
//! `cycle_language_shortcut`) and are edited in the Settings tab.

use tauri_plugin_global_shortcut::{GlobalShortcutExt, ShortcutState};

pub const SHORTCUT_RECORD: &str = "record";
pub const SHORTCUT_CYCLE_LANGUAGE: &str = "cycle-language";

/// Drops the current bindings and binds the shortcuts from settings. An
/// empty combination leaves its action unbound.
pub fn register(app: &tauri::AppHandle) -> Result<(), String> {
	let settings = crate::settings::load();
	let shortcuts = app.global_shortcut();
	shortcuts
		.unregister_all()
		.map_err(|e| format!("cannot unbind shortcuts: {e}"))?;
	let bindings = [
		(SHORTCUT_RECORD, settings.record_shortcut),
		(SHORTCUT_CYCLE_LANGUAGE, settings.cycle_language_shortcut),
	];
	for (id, keys) in bindings {
		let keys = keys.trim();
		if keys.is_empty() {
			continue;
		}
		shortcuts
			.on_shortcut(keys, move |app, _shortcut, event| {
				if event.state() == ShortcutState::Pressed {
					crate::on_shortcut(app, id);
				}
			})
			.map_err(|e| format!("cannot bind {id} to {keys}: {e}"))?;
	}
	Ok(())
}

/// Applies the shortcuts saved in settings (Settings tab).
#[tauri::command]
pub async fn rebind_shortcuts(app: tauri::AppHandle) -> Result<(), String> {
	register(&app)
}
