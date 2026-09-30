//! wtf — local voice-to-text dictation (see DESIGN.md for decisions).

// Skeleton: pipeline seams are wired up incrementally; keep unwired
// functions warning-free until then.
#![allow(dead_code)]

mod app_id;
mod asr;
mod audio;
mod gigaam;
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
			// With start_hidden the settings window stays in the tray at launch;
			// setup runs before the event loop maps the window, so it never
			// flashes. The tray menu and single-instance handler show it.
			if settings::load().start_hidden {
				if let Some(main) = app.get_webview_window("main") {
					let _ = main.hide();
				}
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
			models::check_model_updates,
			inject::copy_to_clipboard,
			asr::list_gpu_devices,
			hotkey::rebind_shortcuts,
			pipeline::unload_model
		])
		.run(tauri::generate_context!())
		.expect("error while running tauri application");
}
