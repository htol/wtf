//! wtf — local voice-to-text dictation (see DESIGN.md for decisions).

// Skeleton: pipeline seams are wired up incrementally; keep unwired
// functions warning-free until then.
#![allow(dead_code)]

mod app_id;
mod asr;
mod audio;
mod history;
mod hotkey;
mod inject;
mod models;
mod settings;
mod tray;

pub fn run() {
	tauri::Builder::default()
		// Dictation daemon lives in the tray: closing the settings window
		// hides it instead of exiting the app (Quit lives in the tray menu).
		.on_window_event(|window, event| {
			if let tauri::WindowEvent::CloseRequested { api, .. } = event {
				api.prevent_close();
				let _ = window.hide();
			}
		})
		.setup(|app| {
			tray::init(app)?;
			// TODO: spawn hotkey::register() on the tokio runtime and wire
			// record-toggle -> audio::Recorder -> asr::Transcriber -> inject::paste
			// -> history::insert; create the overlay window (hidden).
			Ok(())
		})
		.invoke_handler(tauri::generate_handler![
			settings::get_settings,
			settings::set_settings,
			history::list_history
		])
		.run(tauri::generate_context!())
		.expect("error while running tauri application");
}
