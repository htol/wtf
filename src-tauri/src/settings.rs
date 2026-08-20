//! Persistent user settings, stored as JSON in the config directory.

use serde::{Deserialize, Serialize};

use crate::app_id;

/// A named whisper initial prompt (decoder conditioning text; see
/// asr::Transcriber::transcribe).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NamedPrompt {
	pub name: String,
	pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
	/// Whisper language code, or "auto" for detection.
	pub language: String,
	/// GPU device index passed to whisper (`gpu_device`); 0 = first GPU.
	pub gpu_device: i32,
	/// Run inference on GPU at all.
	pub use_gpu: bool,
	/// Path override for the ggml model; None = managed download in models_dir.
	pub model_path: Option<String>,
	/// Chosen model id (see models::MODEL_CHOICES); None = most recent download.
	pub model_id: Option<String>,
	/// Named initial prompts; `active_prompt` names the one in use (None = off).
	pub prompts: Vec<NamedPrompt>,
	pub active_prompt: Option<String>,
	/// Overlay position as fractions of the screen (x, y).
	pub overlay_x: f64,
	pub overlay_y: f64,
	/// Peak amplitude below which a recording counts as silence and never
	/// reaches whisper (hallucination guard); 0 disables the check.
	pub silence_peak: f32,
}

impl Default for Settings {
	fn default() -> Self {
		Self {
			language: "auto".into(),
			gpu_device: 0,
			use_gpu: true,
			model_path: None,
			model_id: None,
			prompts: vec![NamedPrompt {
				name: "ru-en mix".into(),
				text: "Сегодня у нас meeting по архитектуре, я закинул PR и обновил roadmap. "
					.into(),
			}],
			active_prompt: None,
			overlay_x: 0.5,
			// DESIGN.md: initial indicator position = 20% from bottom.
			overlay_y: 0.8,
			silence_peak: 0.1,
		}
	}
}

fn path() -> std::path::PathBuf {
	app_id::config_dir().join("settings.json")
}

pub fn load() -> Settings {
	let Ok(text) = std::fs::read_to_string(path()) else {
		return Settings::default();
	};
	serde_json::from_str(&text).unwrap_or_default()
}

pub fn save(settings: &Settings) -> Result<(), String> {
	let dir = app_id::config_dir();
	std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
	let text = serde_json::to_string_pretty(settings).map_err(|e| e.to_string())?;
	std::fs::write(path(), text).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn get_settings() -> Settings {
	load()
}

#[tauri::command]
pub fn set_settings(app: tauri::AppHandle, settings: Settings) -> Result<(), String> {
	let previous = load();
	save(&settings)?;
	// The transcriber bakes these in at model load; apply them by dropping
	// it — the next dictation reloads with the new values. Without this the
	// old model would stay in VRAM until the next dictation swaps it.
	if settings.model_path != previous.model_path
		|| settings.model_id != previous.model_id
		|| settings.gpu_device != previous.gpu_device
		|| settings.use_gpu != previous.use_gpu
	{
		crate::pipeline::unload_transcriber(&app);
	}
	Ok(())
}

/// Persisted overlay position update from the overlay window's own drag
/// events (fractions of the current monitor).
#[tauri::command]
pub fn set_overlay_position(x: f64, y: f64) -> Result<(), String> {
	let mut settings = load();
	settings.overlay_x = x;
	settings.overlay_y = y;
	save(&settings)
}
