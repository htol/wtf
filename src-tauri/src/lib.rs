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
		use tauri::Manager;
		let hotkeys = match hotkey::register().await {
			Ok(hotkeys) => hotkeys,
			Err(e) => {
				eprintln!("hotkey: portal registration failed: {e}");
				return;
			}
		};
		let hub = std::sync::Arc::new(hotkeys);
		app.manage(hub.clone());
		let app_for_record = app.clone();
		let emit = |id: &str| {
			eprintln!("hotkey activated: {id}");
			if id == hotkey::SHORTCUT_RECORD {
				pipeline::toggle_record(&app_for_record);
			}
			let _ = tauri::Emitter::emit(&app, "shortcut", id.to_string());
		};
		if let Err(e) = hotkey::listen(&hub.globals, emit).await {
			eprintln!("hotkey: activation stream ended: {e}");
		}
	});
}

pub fn run() {
	tauri::Builder::default()
		// Dictation daemon lives in the tray: closing the settings window
		// hides it instead of exiting the app (Quit lives in the tray menu);
		// the overlay is never closable, only shown/hidden by the pipeline.
		.on_window_event(|window, event| {
			if let tauri::WindowEvent::CloseRequested { api, .. } = event {
				api.prevent_close();
				if window.label() != "overlay" {
					let _ = window.hide();
				}
			}
		})
		.setup(|app| {
			use tauri::Manager;
			tray::init(app)?;
			app.manage(pipeline::Dictation::new());
			// Keep the overlay mapped (1x1 invisible) from startup: on Wayland
			// every show() of an unmapped window can activate it and steal
			// focus from the dictation target, while resizes never do.
			pipeline::prime_overlay(app.handle());
			spawn_hotkeys(app.handle().clone());
			Ok(())
		})
		.invoke_handler(tauri::generate_handler![
			settings::get_settings,
			settings::set_settings,
			settings::set_overlay_position,
			history::list_history,
			history::delete_history,
			models::list_models,
			models::download_model,
			models::delete_model,
			models::open_models_dir,
			inject::copy_to_clipboard,
			asr::list_gpu_devices,
			hotkey::rebind_shortcuts
		])
		.run(tauri::generate_context!())
		.expect("error while running tauri application");
}
