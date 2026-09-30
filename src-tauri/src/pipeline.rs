//! Dictation pipeline: `record` hotkey toggle -> capture -> transcribe ->
//! paste into the focused app -> history (DESIGN.md "Pipeline").
//!
//! Engine routing (DESIGN.md, "Engines"): language `ru` -> GigaAM when its
//! model is downloaded; `auto`, `en` and `ru-whisper` (Russian pinned to
//! whisper) -> whisper. `ru` without the GigaAM model falls back to whisper
//! with a one-time download suggestion.
//!
//! `Dictation` is app-managed state. Engines are created lazily on the first
//! transcription and cached, each slot keyed by model path (loading is
//! expensive); both stay loaded across language switches.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use tauri::{Emitter, Manager};

use crate::{asr, audio, history, inject, models, settings};

pub struct Dictation {
	recorder: Mutex<Option<audio::Recorder>>,
	/// Whisper slot: Transcriber::Whisper, keyed by ggml model path.
	whisper: Mutex<Option<(PathBuf, asr::Transcriber)>>,
	/// GigaAM slot: Transcriber::GigaAm, keyed by onnx model path.
	gigaam: Mutex<Option<(PathBuf, asr::Transcriber)>>,
	/// Guard for the one-time-per-session GigaAM download suggestion.
	gigaam_suggested: AtomicBool,
}

impl Dictation {
	pub fn new() -> Self {
		Self {
			recorder: Mutex::new(None),
			whisper: Mutex::new(None),
			gigaam: Mutex::new(None),
			gigaam_suggested: AtomicBool::new(false),
		}
	}
}

/// Handles a `record` activation: starts or stops recording. Stopping runs
/// the rest of the pipeline (transcribe -> paste -> history) on a blocking
/// task so the portal activation stream is never blocked.
pub fn toggle_record(app: &tauri::AppHandle) {
	let state = app.state::<Dictation>();
	let mut slot = state.recorder.lock().unwrap();
	if let Some(recorder) = slot.take() {
		drop(slot); // don't hold the lock while stopping the stream
		crate::tray::set_recording(app, false);
		let _ = app.emit("recording", false);
		let app = app.clone();
		tauri::async_runtime::spawn_blocking(move || {
			// Model load + inference + resample take seconds: all blocking.
			let outcome = recorder
				.stop()
				.map_err(|e| format!("record stop failed: {e}"))
				.and_then(|samples| {
				// Silence guard: whisper hallucinates fluent text on near-silence.
				// Threshold comes from settings; 0 disables the check.
				let threshold = settings::load().silence_peak;
				let peak = audio::peak(&samples);
				if threshold > 0.0 && peak < threshold {
					eprintln!("no speech detected (peak {peak:.3} < {threshold:.3})");
					let _ = app.emit("no-speech", ());
					return Ok(());
				}
				let _ = app.emit("processing", true);
					let result = transcribe_and_paste(&app, &samples);
					let _ = app.emit("processing", false);
					result
				});
			hide_overlay(&app, outcome.is_err());
			if let Err(e) = outcome {
				eprintln!("pipeline failed: {e}");
				let _ = app.emit("pipeline-error", e);
			}
		});
		return;
	}
	match audio::Recorder::start() {
		Ok(recorder) => {
			*slot = Some(recorder);
			crate::tray::set_recording(app, true);
			show_overlay(app);
			spawn_level_ticker(app.clone());
			let _ = app.emit("recording", true);
		}
		Err(e) => {
			eprintln!("record start failed: {e}");
			let _ = app.emit("pipeline-error", format!("record start failed: {e}"));
		}
	}
}

/// Prepares the overlay window without mapping it. GTK3 on Wayland cannot
/// resize a mapped toplevel (`gtk_window_resize` is silently ignored after
/// map — the compositor owns the size), so the overlay is shown/hidden
/// instead of collapsed to 1x1. `set_focusable(false)` must run before the
/// first `show()`; on Wayland first maps can activate the window, so the
/// compositor config must keep it out of focus (Hyprland: `no_initial_focus`).
pub fn prime_overlay(app: &tauri::AppHandle) {
	let Some(window) = app.get_webview_window("overlay") else {
		return;
	};
	let _ = window.set_focusable(false);
	let _ = window.set_always_on_top(true);
}

/// Maps the overlay at its recording size (260x30 from the window config).
/// Positioning is left to the compositor (KWin rule "Remember", Hyprland
/// stores the last floating position): Wayland ignores client-side
/// set_position.
fn show_overlay(app: &tauri::AppHandle) {
	let Some(window) = app.get_webview_window("overlay") else {
		return;
	};
	let _ = window.show();
}

/// Unmaps the overlay; on error keeps it around briefly to show the message.
fn hide_overlay(app: &tauri::AppHandle, failed: bool) {
	let Some(window) = app.get_webview_window("overlay") else {
		return;
	};
	if !failed {
		let _ = window.hide();
		return;
	}
	std::thread::sleep(std::time::Duration::from_millis(2500));
	let _ = window.hide();
}

/// Emits `level` events (~10 Hz) while a recorder is active.
fn spawn_level_ticker(app: tauri::AppHandle) {
	std::thread::spawn(move || loop {
		std::thread::sleep(std::time::Duration::from_millis(100));
		let state = app.state::<Dictation>();
		let guard = state.recorder.lock().unwrap();
		let Some(recorder) = guard.as_ref() else {
			break;
		};
		let _ = app.emit("level", recorder.level());
	});
}

fn transcribe_and_paste(app: &tauri::AppHandle, samples: &[f32]) -> Result<(), String> {
	let settings = settings::load();
	// Two Russian rows in the selector: `ru` prefers GigaAM, `ru-whisper`
	// pins Russian to whisper; both force the language code so the whisper
	// fallback/fixed path transcribes Russian.
	let use_gigaam = settings.language == "ru";
	let language = match settings.language.as_str() {
		"auto" => None,
		"ru-whisper" => Some("ru"),
		code => Some(code),
	};
	let prompt = settings
		.active_prompt
		.as_ref()
		.and_then(|name| settings.prompts.iter().find(|p| &p.name == name))
		.map(|p| p.text.as_str());
	let (text, lang) = if use_gigaam {
		match models::resolve_gigaam(settings.gigaam_model_id.as_deref()) {
			Some(model) => (
				transcribe_gigaam_cached(app, &model, samples)?,
				"ru".to_string(),
			),
			None => {
				// No GigaAM model: whisper handles Russian, and the user gets
				// one nudge per session towards the better engine.
				suggest_gigaam_download(app);
				let model = require_whisper_model(app)?;
				transcribe_whisper_cached(app, &model, samples, Some("ru"), prompt)?
			}
		}
	} else {
		let model = require_whisper_model(app)?;
		transcribe_whisper_cached(app, &model, samples, language, prompt)?
	};
	// History is written before pasting: a paste failure must not lose the
	// transcript. Its error is deferred so a history failure does not block
	// the paste either.
	let history = history::open().and_then(|conn| history::insert(&conn, &text, &lang));
	// The `transcript` event is what refreshes the History tab; it must not
	// be gated on paste success — a failed paste still leaves the new row in
	// the database, and an open History window would otherwise stay stale.
	let _ = app.emit("transcript", &text);
	inject::paste(&text)?;
	history?;
	Ok(())
}

/// Resolves the whisper model or opens the settings window on the model
/// picker instead of failing silently (first run).
fn require_whisper_model(app: &tauri::AppHandle) -> Result<PathBuf, String> {
	let settings = settings::load();
	models::resolve(settings.model_path.as_deref(), settings.model_id.as_deref()).ok_or_else(|| {
		if let Some(window) = app.get_webview_window("main") {
			let _ = window.show();
			let _ = window.set_focus();
		}
		let _ = app.emit("no-model", ());
		"no model available: download one in settings".to_string()
	})
}

/// One-time-per-session desktop notification: Russian dictation could use
/// the GigaAM engine (DESIGN.md "Engines"). Fire-and-forget: portal
/// notification failures only cost the suggestion.
fn suggest_gigaam_download(app: &tauri::AppHandle) {
	let state = app.state::<Dictation>();
	if state.gigaam_suggested.swap(true, Ordering::Relaxed) {
		return;
	}
	let _ = app.emit("gigaam-suggest", ());
	tauri::async_runtime::spawn(async move {
		use ashpd::desktop::notification::NotificationProxy;
		let Ok(proxy) = NotificationProxy::new().await else {
			return;
		};
		let notification = ashpd::desktop::notification::Notification::new("wtf")
			.body("Russian dictation can use the GigaAM engine - download it in Settings, Model section.");
		let _ = proxy.add_notification("wtf-gigaam-suggest", notification).await;
	});
}

/// Drops both cached engines: model weights and whisper state are freed
/// (GPU VRAM + host memory); the next dictation reloads lazily.
pub fn unload_transcriber(app: &tauri::AppHandle) {
	let state = app.state::<Dictation>();
	let whisper = state.whisper.lock().unwrap().take().is_some();
	let gigaam = state.gigaam.lock().unwrap().take().is_some();
	if whisper || gigaam {
		eprintln!("models unloaded");
	}
}

/// Drops only the cached GigaAM engine (its model file changed or was
/// deleted).
pub fn unload_gigaam(app: &tauri::AppHandle) {
	let state = app.state::<Dictation>();
	if state.gigaam.lock().unwrap().take().is_some() {
		eprintln!("gigaam model unloaded");
	}
}

#[tauri::command]
pub fn unload_model(app: tauri::AppHandle) {
	unload_transcriber(&app);
}

fn transcribe_whisper_cached(
	app: &tauri::AppHandle,
	model: &std::path::Path,
	samples: &[f32],
	language: Option<&str>,
	initial_prompt: Option<&str>,
) -> Result<(String, String), String> {
	let settings = settings::load();
	let state = app.state::<Dictation>();
	let mut cached = state.whisper.lock().unwrap();
	if !cached.as_ref().is_some_and(|(path, _)| path == model) {
		eprintln!("loading model {}...", model.display());
		*cached = Some((
			model.to_path_buf(),
			asr::Transcriber::whisper(
				&model.to_string_lossy(),
				settings.gpu_device,
				settings.use_gpu,
			)?,
		));
	}
	cached
		.as_mut()
		.expect("transcriber was just stored")
		.1
		.transcribe(samples, language, initial_prompt)
}

fn transcribe_gigaam_cached(
	app: &tauri::AppHandle,
	model: &std::path::Path,
	samples: &[f32],
) -> Result<String, String> {
	let state = app.state::<Dictation>();
	let mut cached = state.gigaam.lock().unwrap();
	if !cached.as_ref().is_some_and(|(path, _)| path == model) {
		eprintln!("loading gigaam model {}...", model.display());
		*cached = Some((
			model.to_path_buf(),
			asr::Transcriber::GigaAm(crate::gigaam::GigaAm::new(model)?),
		));
	}
	let (text, _lang) = cached
		.as_mut()
		.expect("transcriber was just stored")
		.1
		.transcribe(samples, Some("ru"), None)?;
	Ok(text)
}
