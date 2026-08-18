//! Model management: download into the models dir on first run, or use a
//! manual path override from settings.
//!
//! Default recommendation: ggml-large-v3-turbo q5_0 from
//! https://huggingface.co/ggerganov/whisper.cpp
//! (multilingual models, see DESIGN.md "Models").

use std::path::PathBuf;

use futures_util::StreamExt;
use tauri::Emitter;

use crate::app_id;

const HF_BASE: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";

/// Candidate models for the first-run picker.
pub const MODEL_CHOICES: &[(&str, &str)] = &[
	// (id, huggingface ggml file name)
	("tiny", "ggml-tiny.bin"),
	("base", "ggml-base.bin"),
	("small", "ggml-small.bin"),
	("medium", "ggml-medium.bin"),
	("large-v3-turbo-q8_0", "ggml-large-v3-turbo-q8_0.bin"),
	("large-v3-turbo", "ggml-large-v3-turbo.bin"),
	("large-v3", "ggml-large-v3.bin"),
];

/// Resolves the model file to use: manual override from settings, else the
/// chosen model (when its file is installed), else the most recent download,
/// else None (first-run picker).
pub fn resolve(manual_path: Option<&str>, model_id: Option<&str>) -> Option<PathBuf> {
	if let Some(p) = manual_path {
		return Some(PathBuf::from(p));
	}
	let dir = app_id::models_dir();
	if let Some(id) = model_id {
		let file = MODEL_CHOICES
			.iter()
			.find(|&&(choice, _)| choice == id)
			.map(|&(_, file)| file);
		if let Some(file) = file {
			let path = dir.join(file);
			if path.is_file() {
				return Some(path);
			}
		}
	}
	let latest = std::fs::read_dir(&dir)
		.ok()?
		.filter_map(|e| e.ok())
		.filter(|e| e.path().extension().is_some_and(|x| x == "bin"))
		.max_by_key(|e| e.metadata().ok().and_then(|m| m.modified().ok()));
	latest.map(|e| e.path())
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ModelInfo {
	pub id: &'static str,
	pub file: &'static str,
	pub installed: bool,
	/// Size on disk in bytes when installed.
	pub size_bytes: Option<u64>,
	pub active: bool,
}

#[tauri::command]
pub fn list_models(manual_path: Option<String>, model_id: Option<String>) -> Vec<ModelInfo> {
	let dir = app_id::models_dir();
	let active = resolve(manual_path.as_deref(), model_id.as_deref());
	MODEL_CHOICES
		.iter()
		.map(|&(id, file)| {
			let path = dir.join(file);
			let installed = path.is_file();
			let size_bytes = installed
				.then(|| std::fs::metadata(&path).ok())
				.flatten()
				.map(|m| m.len());
			ModelInfo {
				id,
				file,
				installed,
				size_bytes,
				active: active.as_deref() == Some(path.as_path()),
			}
		})
		.collect()
}

/// Progress event payload for `download_model`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DownloadProgress {
	pub id: &'static str,
	pub downloaded: u64,
	pub total: Option<u64>,
	pub done: bool,
}

/// Minimum bytes between two progress events (large models otherwise emit
/// tens of thousands of them).
const PROGRESS_STEP: u64 = 32 * 1024 * 1024;

#[tauri::command]
pub async fn download_model(app: tauri::AppHandle, model_id: String) -> Result<(), String> {
	let (id, file) = MODEL_CHOICES
		.iter()
		.find(|(id, _)| *id == model_id)
		.map(|&(id, file)| (id, file))
		.ok_or_else(|| format!("unknown model: {model_id}"))?;
	let dir = app_id::models_dir();
	std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
	let final_path = dir.join(file);
	let part_path = dir.join(format!("{file}.part"));

	let response = reqwest::get(format!("{HF_BASE}/{file}"))
		.await
		.map_err(|e| format!("download request failed: {e}"))?
		.error_for_status()
		.map_err(|e| format!("huggingface returned an error: {e}"))?;
	let total = response.content_length();

	let emit = |downloaded: u64, done: bool| {
		let _ = app.emit(
			"model-download",
			DownloadProgress {
				id,
				downloaded,
				total,
				done,
			},
		);
	};

	let mut part = tokio::fs::File::create(&part_path)
		.await
		.map_err(|e| format!("cannot create {}: {e}", part_path.display()))?;
	use tokio::io::AsyncWriteExt;
	let mut downloaded: u64 = 0;
	let mut last_emitted: u64 = 0;
	let mut stream = response.bytes_stream();
	while let Some(chunk) = stream.next().await {
		let chunk = chunk.map_err(|e| format!("download interrupted: {e}"))?;
		part
			.write_all(&chunk)
			.await
			.map_err(|e| format!("write failed: {e}"))?;
		downloaded += chunk.len() as u64;
		if downloaded - last_emitted >= PROGRESS_STEP {
			last_emitted = downloaded;
			emit(downloaded, false);
		}
	}
	part.flush().await.map_err(|e| e.to_string())?;
	drop(part);
	std::fs::rename(&part_path, &final_path)
		.map_err(|e| format!("cannot finalize download: {e}"))?;
	emit(downloaded, true);
	Ok(())
}

/// Removes a downloaded model file (and its stale partial download, if any)
/// from the models dir. Managed files only: paths come from MODEL_CHOICES.
#[tauri::command]
pub fn delete_model(model_id: String) -> Result<(), String> {
	let (_, file) = MODEL_CHOICES
		.iter()
		.find(|(id, _)| *id == model_id)
		.map(|&(id, file)| (id, file))
		.ok_or_else(|| format!("unknown model: {model_id}"))?;
	let dir = app_id::models_dir();
	let path = dir.join(file);
	if path.is_file() {
		std::fs::remove_file(&path).map_err(|e| format!("cannot delete {}: {e}", path.display()))?;
	}
	let _ = std::fs::remove_file(dir.join(format!("{file}.part")));
	Ok(())
}

/// Opens the models directory in the desktop file manager.
#[tauri::command]
pub fn open_models_dir() -> Result<(), String> {
	let dir = app_id::models_dir();
	std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
	std::process::Command::new("xdg-open")
		.arg(&dir)
		.spawn()
		.map_err(|e| format!("xdg-open: {e}"))?;
	Ok(())
}
