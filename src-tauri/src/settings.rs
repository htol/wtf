//! Persistent user settings, stored as JSON in the config directory.

use serde::{Deserialize, Serialize};

use crate::app_id;

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
	/// Overlay position as fractions of the screen (x, y).
	pub overlay_x: f64,
	pub overlay_y: f64,
}

impl Default for Settings {
	fn default() -> Self {
		Self {
			language: "auto".into(),
			gpu_device: 0,
			use_gpu: true,
			model_path: None,
			overlay_x: 0.5,
			// DESIGN.md: initial indicator position = 20% from bottom.
			overlay_y: 0.8,
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
pub fn set_settings(settings: Settings) -> Result<(), String> {
	save(&settings)
}
