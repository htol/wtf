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
mod pipeline;
mod settings;
mod tray;

/// Registers the global shortcuts and forwards every activation into the
/// app as a `shortcut` event (payload: shortcut id). Runs on the tauri
/// (tokio) runtime for the lifetime of the daemon.
fn spawn_hotkeys(app: tauri::AppHandle) {
	tauri::async_runtime::spawn(async move {
		let globals = match hotkey::register().await {
			Ok(globals) => globals,
			Err(e) => {
				eprintln!("hotkey: portal registration failed: {e}");
				return;
			}
		};
		let app_for_record = app.clone();
		let emit = |id: &str| {
			eprintln!("hotkey activated: {id}");
			if id == hotkey::SHORTCUT_RECORD {
				pipeline::toggle_record(&app_for_record);
			}
			let _ = tauri::Emitter::emit(&app, "shortcut", id.to_string());
		};
		if let Err(e) = hotkey::listen(&globals, emit).await {
			eprintln!("hotkey: activation stream ended: {e}");
		}
	});
}

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
			use tauri::Manager;
			tray::init(app)?;
			app.manage(pipeline::Dictation::new());
			spawn_hotkeys(app.handle().clone());
			Ok(())
		})
		.invoke_handler(tauri::generate_handler![
			settings::get_settings,
			settings::set_settings,
			history::list_history,
			history::delete_history,
			models::list_models,
			models::download_model,
			models::delete_model,
			models::open_models_dir,
			inject::copy_to_clipboard,
			asr::list_gpu_devices
		])
		.run(tauri::generate_context!())
		.expect("error while running tauri application");
}
