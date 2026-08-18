//! Dictation pipeline: `record` hotkey toggle -> capture -> transcribe ->
//! paste into the focused app -> history (DESIGN.md "Pipeline").
//!
//! `Dictation` is app-managed state. The transcriber is created lazily on
//! the first transcription and cached, keyed by model path (loading a model
//! is expensive).

use std::path::PathBuf;
use std::sync::Mutex;

use tauri::{Emitter, Manager};

use crate::{asr, audio, history, inject, models, settings};

pub struct Dictation {
	recorder: Mutex<Option<audio::Recorder>>,
	transcriber: Mutex<Option<(PathBuf, asr::Transcriber)>>,
}

impl Dictation {
	pub fn new() -> Self {
		Self {
			recorder: Mutex::new(None),
			transcriber: Mutex::new(None),
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
		let _ = app.emit("recording", false);
		let app = app.clone();
		tauri::async_runtime::spawn_blocking(move || {
			// Model load + inference + resample take seconds: all blocking.
			let outcome = recorder
				.stop()
				.map_err(|e| format!("record stop failed: {e}"))
				.and_then(|samples| {
				let _ = app.emit("processing", true);
				let result = transcribe_and_paste(&app, &samples);
				let _ = app.emit("processing", false);
				result
			});
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
			let _ = app.emit("recording", true);
		}
		Err(e) => {
			eprintln!("record start failed: {e}");
			let _ = app.emit("pipeline-error", format!("record start failed: {e}"));
		}
	}
}

fn transcribe_and_paste(app: &tauri::AppHandle, samples: &[f32]) -> Result<(), String> {
	let settings = settings::load();
	let Some(model) = models::resolve(settings.model_path.as_deref(), settings.model_id.as_deref()) else {
		// First run: no model yet. Open the settings window on the model
		// picker instead of failing silently.
		if let Some(window) = app.get_webview_window("main") {
			let _ = window.show();
			let _ = window.set_focus();
		}
		let _ = app.emit("no-model", ());
		return Err("no model available: download one in settings".into());
	};
	let language = match settings.language.as_str() {
		"auto" => None,
		code => Some(code),
	};
	let text = transcribe_cached(app, &model, samples, language)?;
	inject::paste(&text)?;
	let conn = history::open()?;
	history::insert(&conn, &text, language.unwrap_or("auto"))?;
	let _ = app.emit("transcript", &text);
	Ok(())
}

fn transcribe_cached(
	app: &tauri::AppHandle,
	model: &std::path::Path,
	samples: &[f32],
	language: Option<&str>,
) -> Result<String, String> {
	let settings = settings::load();
	let state = app.state::<Dictation>();
	let mut cached = state.transcriber.lock().unwrap();
	if !cached.as_ref().is_some_and(|(path, _)| path == model) {
		eprintln!("loading model {}...", model.display());
		*cached = Some((
			model.to_path_buf(),
			asr::Transcriber::new(
				&model.to_string_lossy(),
				settings.gpu_device,
				settings.use_gpu,
			)?,
		));
	}
	cached
		.as_ref()
		.expect("transcriber was just stored")
		.1
		.transcribe(samples, language)
}
