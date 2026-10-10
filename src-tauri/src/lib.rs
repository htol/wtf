//! wtf — local voice-to-text dictation (see DESIGN.md for decisions).

// Skeleton: pipeline seams are wired up incrementally; keep unwired
// functions warning-free until then.
#![allow(dead_code)]

mod app_id;
mod asr;
mod audio;
mod gigaam;
mod history;
#[cfg(target_os = "linux")]
mod hotkey;
#[cfg(target_os = "macos")]
#[path = "hotkey_macos.rs"]
mod hotkey;
#[cfg(target_os = "linux")]
mod inject;
#[cfg(target_os = "macos")]
#[path = "inject_macos.rs"]
mod inject;
mod models;
mod pipeline;
mod qwen;
mod settings;
mod tray;
mod vad;

/// Handles one global shortcut activation: `record` toggles the pipeline,
/// and every activation reaches the frontend as a `shortcut` event
/// (payload: shortcut id).
fn on_shortcut(app: &tauri::AppHandle, id: &str) {
	eprintln!("hotkey activated: {id}");
	if id == hotkey::SHORTCUT_RECORD {
		pipeline::toggle_record(app);
	}
	let _ = tauri::Emitter::emit(app, "shortcut", id.to_string());
}

/// Registers the shortcuts from settings; activations arrive on the main
/// thread through the global-shortcut plugin.
#[cfg(target_os = "macos")]
fn spawn_hotkeys(app: tauri::AppHandle) {
	let registered = app
		.plugin(tauri_plugin_global_shortcut::Builder::new().build())
		.map_err(|e| e.to_string())
		.and_then(|()| hotkey::register(&app));
	if let Err(e) = registered {
		eprintln!("hotkey: registration failed: {e}");
	}
}

/// Registers the global shortcuts and forwards every activation to
/// `on_shortcut`. Runs on the tauri (tokio) runtime for the lifetime of the
/// daemon.
#[cfg(target_os = "linux")]
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
		let emit = |id: &str| on_shortcut(&app, id);
		if let Err(e) = hotkey::listen(&hub.globals, emit).await {
			eprintln!("hotkey: activation stream ended: {e}");
		}
	});
}

pub fn run() {
	tauri::Builder::default()
		// Plasma session restore relaunches the app at login even though
		// app-wtf.service already started it: restore's dedup only knows XDG
		// autostart, not systemd user units (see README, Runtime dependencies).
		// A second copy would race for the tray, GlobalShortcuts and the audio
		// device, so it exits and surfaces the existing window instead.
		.plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
			use tauri::Manager;
			if let Some(main) = app.get_webview_window("main") {
				let _ = main.show();
				let _ = main.set_focus();
			}
		}))
		// Dictation daemon lives in the tray: closing the settings window
		// hides it instead of exiting the app (Quit lives in the tray menu);
		// the overlay is only shown/hidden by the pipeline — but it must obey
		// close requests. Nothing closes it mid-session; the only realistic
		// source of a close request is the session manager at logout.
		.on_window_event(|window, event| {
			if let tauri::WindowEvent::CloseRequested { api, .. } = event {
				if window.label() == "overlay" {
					return;
				}
				api.prevent_close();
				let _ = window.hide();
			}
		})
		.setup(|app| {
			use tauri::Manager;
			tray::init(app)?;
			app.manage(pipeline::Dictation::new());
		// The overlay is unmapped while idle and only mapped by the pipeline
		// while recording (GTK3 cannot resize a mapped Wayland toplevel, so
		// show/hide replaced the old collapse-to-1x1). Keeping it from stealing
		// focus on map is the compositor's job (Hyprland: no_initial_focus rule).
		pipeline::prime_overlay(app.handle());
			spawn_hotkeys(app.handle().clone());
			// Ask for the paste permission at launch rather than after the
			// first dictation has already failed to paste.
			#[cfg(target_os = "macos")]
			inject::request_accessibility();
			// With start_hidden the settings window stays in the tray at launch;
			// setup runs before the event loop maps the window, so it never
			// flashes. The tray menu and single-instance handler show it.
			let settings = settings::load();
			if settings.start_hidden {
				if let Some(main) = app.get_webview_window("main") {
					let _ = main.hide();
				}
			}
			if settings.preload_model {
				let app = app.handle().clone();
				// Loading takes seconds: keep it off the main thread.
				tauri::async_runtime::spawn_blocking(move || {
					if let Err(e) = pipeline::preload(&app) {
						eprintln!("model preload failed: {e}");
					}
				});
			}
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
			models::list_gigaam_models,
			models::download_gigaam_model,
			models::delete_gigaam_model,
			models::list_qwen_models,
			models::download_qwen_model,
			models::delete_qwen_model,
			models::check_model_updates,
			inject::copy_to_clipboard,
			audio::list_input_devices,
			asr::list_gpu_devices,
			qwen::list_qwen_devices,
			hotkey::rebind_shortcuts,
			pipeline::unload_model
		])
		.run(tauri::generate_context!())
		.expect("error while running tauri application");
}
