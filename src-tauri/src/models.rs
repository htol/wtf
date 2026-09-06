//! Model management: download into the models dir on first run, or use a
//! manual path override from settings.
//!
//! Default recommendation: ggml-large-v3-turbo q5_0 from
//! https://huggingface.co/ggerganov/whisper.cpp
//! (multilingual models, see DESIGN.md "Models"). The GigaAM engine has its
//! own catalog and storage subdir (DESIGN.md, "Engines").

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
	download_with_progress(&app, id, &format!("{HF_BASE}/{file}"), &dir, file).await
}

/// Streams `url` into `dir/{file}.part`, emitting `model-download` progress
/// events under `id`, and renames it into place on completion.
async fn download_with_progress(
	app: &tauri::AppHandle,
	id: &'static str,
	url: &str,
	dir: &std::path::Path,
	file: &str,
) -> Result<(), String> {
	let final_path = dir.join(file);
	let part_path = dir.join(format!("{file}.part"));

	let response = reqwest::get(url)
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

// --- GigaAM (Russian engine; see DESIGN.md "Engines") ---

/// Pinned-revision base URL for the GigaAM catalog downloads.
const GIGAAM_HF_BASE: &str =
	"https://huggingface.co/istupakov/gigaam-v3-onnx/resolve/322c3b29492673eb7d0b434bfa9dfb8653e34d02";

/// Vocab shared by both GigaAM catalog entries.
pub const GIGAAM_VOCAB_FILE: &str = "v3_e2e_ctc_vocab.txt";

/// A GigaAM catalog entry.
pub struct GigaamChoice {
	pub id: &'static str,
	pub file: &'static str,
	/// One-line UI description.
	pub note: &'static str,
	/// Download size on huggingface (shown before install).
	pub approx_bytes: u64,
}

pub const GIGAAM_CHOICES: &[GigaamChoice] = &[
	GigaamChoice {
		id: "gigaam-v3-e2e-ctc",
		file: "v3_e2e_ctc.onnx",
		// fp32 measured faster than int8 on this machine (QLinearConv
		// overhead) — spike branch `spike/gigaam`.
		note: "fp32 — recommended: fastest on CPU",
		approx_bytes: 885_950_079,
	},
	GigaamChoice {
		id: "gigaam-v3-e2e-ctc-int8",
		file: "v3_e2e_ctc.int8.onnx",
		note: "int8 — compact",
		approx_bytes: 224_893_347,
	},
];

/// GigaAM models live in their own subdir of the models dir.
pub fn gigaam_dir() -> PathBuf {
	app_id::models_dir().join("gigaam")
}

/// Resolves the GigaAM model file: the chosen id when installed, else the
/// most recent `.onnx` download, else None (engine not available).
pub fn resolve_gigaam(model_id: Option<&str>) -> Option<PathBuf> {
	let dir = gigaam_dir();
	if let Some(id) = model_id {
		if let Some(choice) = GIGAAM_CHOICES.iter().find(|c| c.id == id) {
			let path = dir.join(choice.file);
			if path.is_file() {
				return Some(path);
			}
		}
	}
	std::fs::read_dir(&dir)
		.ok()?
		.filter_map(|e| e.ok())
		.filter(|e| e.path().extension().is_some_and(|x| x == "onnx"))
		.max_by_key(|e| e.metadata().ok().and_then(|m| m.modified().ok()))
		.map(|e| e.path())
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct GigaamModelInfo {
	pub id: &'static str,
	pub file: &'static str,
	pub note: &'static str,
	pub installed: bool,
	/// Size on disk in bytes when installed.
	pub size_bytes: Option<u64>,
	/// Download size (shown before install).
	pub approx_bytes: u64,
	pub active: bool,
}

#[tauri::command]
pub fn list_gigaam_models(gigaam_model_id: Option<String>) -> Vec<GigaamModelInfo> {
	let dir = gigaam_dir();
	let active = resolve_gigaam(gigaam_model_id.as_deref());
	GIGAAM_CHOICES
		.iter()
		.map(|choice| {
			let path = dir.join(choice.file);
			let installed = path.is_file();
			let size_bytes = installed
				.then(|| std::fs::metadata(&path).ok())
				.flatten()
				.map(|m| m.len());
			GigaamModelInfo {
				id: choice.id,
				file: choice.file,
				note: choice.note,
				installed,
				size_bytes,
				approx_bytes: choice.approx_bytes,
				active: active.as_deref() == Some(path.as_path()),
			}
		})
		.collect()
}

#[tauri::command]
pub async fn download_gigaam_model(app: tauri::AppHandle, model_id: String) -> Result<(), String> {
	let choice = GIGAAM_CHOICES
		.iter()
		.find(|c| c.id == model_id)
		.ok_or_else(|| format!("unknown gigaam model: {model_id}"))?;
	let dir = gigaam_dir();
	std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
	// The vocab is shared by both entries and tiny (2 KB): fetch silently
	// when missing.
	let vocab = dir.join(GIGAAM_VOCAB_FILE);
	if !vocab.is_file() {
		download_file(&format!("{GIGAAM_HF_BASE}/{GIGAAM_VOCAB_FILE}"), &vocab).await?;
	}
	download_with_progress(
		&app,
		choice.id,
		&format!("{GIGAAM_HF_BASE}/{}", choice.file),
		&dir,
		choice.file,
	)
	.await
}

/// Small file download without progress events (vocab).
async fn download_file(url: &str, dest: &std::path::Path) -> Result<(), String> {
	let response = reqwest::get(url)
		.await
		.map_err(|e| format!("download request failed: {e}"))?
		.error_for_status()
		.map_err(|e| format!("huggingface returned an error: {e}"))?;
	let bytes = response.bytes().await.map_err(|e| format!("download interrupted: {e}"))?;
	tokio::fs::write(dest, &bytes).await.map_err(|e| format!("write failed: {e}"))
}

/// Removes a downloaded GigaAM model (and the shared vocab when no model
/// remains). Drops the cached GigaAM engine so deleted weights leave memory.
#[tauri::command]
pub fn delete_gigaam_model(app: tauri::AppHandle, model_id: String) -> Result<(), String> {
	let choice = GIGAAM_CHOICES
		.iter()
		.find(|c| c.id == model_id)
		.ok_or_else(|| format!("unknown gigaam model: {model_id}"))?;
	let dir = gigaam_dir();
	let path = dir.join(choice.file);
	if path.is_file() {
		std::fs::remove_file(&path).map_err(|e| format!("cannot delete {}: {e}", path.display()))?;
	}
	let _ = std::fs::remove_file(dir.join(format!("{}.part", choice.file)));
	let models_left = std::fs::read_dir(&dir)
		.map(|entries| {
			entries
				.filter_map(|e| e.ok())
				.filter(|e| e.path().extension().is_some_and(|x| x == "onnx"))
				.count()
		})
		.unwrap_or(0);
	if models_left == 0 {
		let _ = std::fs::remove_file(dir.join(GIGAAM_VOCAB_FILE));
	}
	crate::pipeline::unload_gigaam(&app);
	Ok(())
}

/// Removes a downloaded model file (and its stale partial download, if any)
/// from the models dir. Managed files only: paths come from MODEL_CHOICES.
#[tauri::command]
pub fn delete_model(app: tauri::AppHandle, model_id: String) -> Result<(), String> {
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
	// The deleted file may be the loaded model: drop the transcriber so its
	// weights leave memory and the next dictation resolves a fresh model.
	crate::pipeline::unload_transcriber(&app);
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
