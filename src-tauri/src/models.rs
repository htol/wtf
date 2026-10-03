//! Model management: download into the models dir on first run, or use a
//! manual path override from settings.
//!
//! Default recommendation: ggml-large-v3-turbo q5_0 from
//! https://huggingface.co/ggerganov/whisper.cpp
//! (multilingual models, see DESIGN.md "Models"). The GigaAM and Qwen3-ASR
//! engines have their own catalogs and storage subdirs (DESIGN.md,
//! "Engines").

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
	download_span(app, id, url, dir, file, 0, None, true).await.map(|_| ())
}

/// One file of a download reported under `id`. `offset` is what earlier
/// files of the same download already fetched and `total` the size of the
/// whole set (None = this file's own length); `last` marks the file whose
/// completion finishes the download. Returns the bytes fetched.
#[allow(clippy::too_many_arguments)]
async fn download_span(
	app: &tauri::AppHandle,
	id: &'static str,
	url: &str,
	dir: &std::path::Path,
	file: &str,
	offset: u64,
	total: Option<u64>,
	last: bool,
) -> Result<u64, String> {
	let final_path = dir.join(file);
	let part_path = dir.join(format!("{file}.part"));

	let response = reqwest::get(url)
		.await
		.map_err(|e| format!("download request failed: {e}"))?
		.error_for_status()
		.map_err(|e| format!("huggingface returned an error: {e}"))?;
	let total = total.or(response.content_length());

	let emit = |downloaded: u64, done: bool| {
		let _ = app.emit(
			"model-download",
			DownloadProgress {
				id,
				downloaded: offset + downloaded,
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
	if last {
		emit(downloaded, true);
	}
	Ok(downloaded)
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

/// Small file download without progress events (vocab, llama.cpp archive).
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

// --- Qwen3-ASR (multilingual engine; see DESIGN.md "Engines") ---

/// A Qwen3-ASR catalog entry: one HF GGUF repo at a pinned revision, stored
/// in its own subdir (named by `id`) of `qwen_dir()`.
pub struct QwenChoice {
	pub id: &'static str,
	/// `{owner}/{repo}` on huggingface.
	repo: &'static str,
	revision: &'static str,
	/// The language model and its audio encoder ("mmproj" in llama.cpp).
	model: &'static str,
	mmproj: &'static str,
	/// One-line UI description.
	pub note: &'static str,
	/// Download size of both files on huggingface (shown before install,
	/// and the progress total).
	pub approx_bytes: u64,
}

impl QwenChoice {
	fn files(&self) -> [&'static str; 2] {
		[self.mmproj, self.model]
	}

	fn installed(&self) -> bool {
		let dir = qwen_dir().join(self.id);
		self.files().iter().all(|file| dir.join(file).is_file())
	}
}

pub const QWEN_CHOICES: &[QwenChoice] = &[
	QwenChoice {
		id: "qwen3-asr-0.6b",
		repo: "ggml-org/Qwen3-ASR-0.6B-GGUF",
		revision: "928ab958557df9aa2ef1c93e0e83c7ad0933fae2",
		model: "Qwen3-ASR-0.6B-Q8_0.gguf",
		mmproj: "mmproj-Qwen3-ASR-0.6B-Q8_0.gguf",
		note: "0.6B Q8 — faster",
		approx_bytes: 1_019_141_728,
	},
	QwenChoice {
		id: "qwen3-asr-1.7b",
		repo: "ggml-org/Qwen3-ASR-1.7B-GGUF",
		revision: "36a678687ba7d07a74ca70ccb0e36902e005fb80",
		model: "Qwen3-ASR-1.7B-Q8_0.gguf",
		mmproj: "mmproj-Qwen3-ASR-1.7B-Q8_0.gguf",
		note: "1.7B Q8 — more accurate",
		approx_bytes: 2_520_744_288,
	},
];

/// Qwen3-ASR models live in their own subdir of the models dir.
pub fn qwen_dir() -> PathBuf {
	app_id::models_dir().join("qwen")
}

/// Resolves the Qwen3-ASR model dir: the chosen id when installed, else the
/// most recently downloaded entry, else None (engine not available).
pub fn resolve_qwen(model_id: Option<&str>) -> Option<PathBuf> {
	QWEN_CHOICES
		.iter()
		.filter(|c| Some(c.id) == model_id)
		.find(|c| c.installed())
		.or_else(|| {
			QWEN_CHOICES.iter().filter(|c| c.installed()).max_by_key(|c| {
				let model = qwen_dir().join(c.id).join(c.model);
				model.metadata().ok().and_then(|m| m.modified().ok())
			})
		})
		.map(|c| qwen_dir().join(c.id))
}

/// The (model, mmproj) files of the catalog entry stored in `dir`.
pub fn qwen_files(dir: &std::path::Path) -> Option<(PathBuf, PathBuf)> {
	let choice = QWEN_CHOICES.iter().find(|c| dir.file_name().is_some_and(|name| name == c.id))?;
	Some((dir.join(choice.model), dir.join(choice.mmproj)))
}

#[tauri::command]
pub fn list_qwen_models(qwen_model_id: Option<String>) -> Vec<GigaamModelInfo> {
	let active = resolve_qwen(qwen_model_id.as_deref());
	QWEN_CHOICES
		.iter()
		.map(|choice| {
			let dir = qwen_dir().join(choice.id);
			let installed = choice.installed();
			let size_bytes = installed.then(|| {
				choice
					.files()
					.iter()
					.filter_map(|file| std::fs::metadata(dir.join(file)).ok())
					.map(|m| m.len())
					.sum()
			});
			GigaamModelInfo {
				id: choice.id,
				file: choice.id,
				note: choice.note,
				installed,
				size_bytes,
				approx_bytes: choice.approx_bytes,
				active: active.as_deref() == Some(dir.as_path()),
			}
		})
		.collect()
}

#[tauri::command]
pub async fn download_qwen_model(app: tauri::AppHandle, model_id: String) -> Result<(), String> {
	let choice = QWEN_CHOICES
		.iter()
		.find(|c| c.id == model_id)
		.ok_or_else(|| format!("unknown qwen model: {model_id}"))?;
	let dir = qwen_dir().join(choice.id);
	std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
	// A re-download replaces files under the running server: stop it.
	crate::pipeline::unload_qwen(&app);
	// The engine's runtime is small next to any model: fetch silently when
	// missing.
	ensure_llama_runtime().await?;
	let mut offset = 0;
	for (i, file) in choice.files().iter().enumerate() {
		let url = format!("https://huggingface.co/{}/resolve/{}/{file}", choice.repo, choice.revision);
		let last = i + 1 == choice.files().len();
		offset +=
			download_span(&app, choice.id, &url, &dir, file, offset, Some(choice.approx_bytes), last)
				.await?;
	}
	Ok(())
}

/// Removes a downloaded Qwen3-ASR model dir and drops the cached engine so
/// its server exits. Managed dirs only: names come from QWEN_CHOICES.
#[tauri::command]
pub fn delete_qwen_model(app: tauri::AppHandle, model_id: String) -> Result<(), String> {
	let choice = QWEN_CHOICES
		.iter()
		.find(|c| c.id == model_id)
		.ok_or_else(|| format!("unknown qwen model: {model_id}"))?;
	let dir = qwen_dir().join(choice.id);
	if dir.is_dir() {
		std::fs::remove_dir_all(&dir).map_err(|e| format!("cannot delete {}: {e}", dir.display()))?;
	}
	crate::pipeline::unload_qwen(&app);
	Ok(())
}

// --- llama.cpp runtime (runs Qwen3-ASR; see DESIGN.md "Engines") ---

/// Pinned llama.cpp release: its prebuilt Vulkan build is unpacked under
/// the data dir and run as `llama-server` (see qwen.rs).
const LLAMA_BUILD: &str = "b11381";

/// sha256 of that release's `bin-ubuntu-vulkan-x64` archive.
const LLAMA_ARCHIVE_SHA256: &str =
	"6f93bf4138e9ad6a380ff605be1fd5108a66ec1977218eb74613f90634e18c64";

fn llama_dir() -> PathBuf {
	app_id::data_dir().join("llama")
}

/// The installed `llama-server` of the pinned build, if any.
pub fn llama_server() -> Option<PathBuf> {
	let path = llama_dir().join(format!("llama-{LLAMA_BUILD}")).join("llama-server");
	path.is_file().then_some(path)
}

/// Downloads and unpacks the pinned llama.cpp build unless it is already
/// installed. The archive holds one `llama-{build}` directory.
pub async fn ensure_llama_runtime() -> Result<PathBuf, String> {
	if let Some(path) = llama_server() {
		return Ok(path);
	}
	let dir = llama_dir();
	std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
	let archive = dir.join(format!("llama-{LLAMA_BUILD}.tar.gz"));
	let url = format!(
		"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_BUILD}/llama-{LLAMA_BUILD}-bin-ubuntu-vulkan-x64.tar.gz"
	);
	download_file(&url, &archive).await?;
	let unpacked = sha256_file(&archive).and_then(|hash| {
		if hash != LLAMA_ARCHIVE_SHA256 {
			return Err(format!("llama.cpp archive has an unexpected sha256: {hash}"));
		}
		let status = std::process::Command::new("tar")
			.arg("-xzf")
			.arg(&archive)
			.arg("-C")
			.arg(&dir)
			.status()
			.map_err(|e| format!("tar: {e}"))?;
		status.success().then_some(()).ok_or_else(|| format!("tar failed: {status}"))
	});
	let _ = std::fs::remove_file(&archive);
	unpacked?;
	llama_server().ok_or_else(|| "llama.cpp archive holds no llama-server".to_string())
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

// --- Update check (installed models vs upstream; DESIGN.md "Models") ---

/// Tree API of the whisper catalog: tracks `main`, so files there can be
/// replaced upstream after a download. LFS entries carry `lfs.oid`, the
/// sha256 of the file content.
const HF_TREE: &str = "https://huggingface.co/api/models/ggerganov/whisper.cpp/tree/main";

/// Tree API of the GigaAM catalog, at the same pinned revision as
/// `GIGAAM_HF_BASE` above (keep the two in sync). The revision is immutable,
/// so a mismatch means the local file predates the current pin or is
/// corrupt.
const GIGAAM_HF_TREE: &str =
	"https://huggingface.co/api/models/istupakov/gigaam-v3-onnx/tree/322c3b29492673eb7d0b434bfa9dfb8653e34d02";

/// HF tree-API entry, reduced to the fields the update check needs.
#[derive(serde::Deserialize)]
struct TreeEntry {
	path: String,
	lfs: Option<TreeLfs>,
}

/// LFS details of a tree entry; `oid` is the sha256 of the content.
#[derive(serde::Deserialize)]
struct TreeLfs {
	oid: String,
}

/// Fetches a tree-API listing and maps file name -> upstream sha256.
/// Non-LFS entries carry no content hash and are skipped.
async fn fetch_upstream_hashes(
	url: &str,
) -> Result<std::collections::HashMap<String, String>, String> {
	let bytes = reqwest::get(url)
		.await
		.map_err(|e| format!("update check request failed: {e}"))?
		.error_for_status()
		.map_err(|e| format!("huggingface returned an error: {e}"))?
		.bytes()
		.await
		.map_err(|e| format!("update check interrupted: {e}"))?;
	let entries: Vec<TreeEntry> = serde_json::from_slice(&bytes)
		.map_err(|e| format!("cannot parse huggingface response: {e}"))?;
	Ok(entries
		.into_iter()
		.filter_map(|e| e.lfs.map(|lfs| (e.path, lfs.oid)))
		.collect())
}

/// Streaming sha256 of a local file, hex-encoded.
fn sha256_file(path: &std::path::Path) -> Result<String, String> {
	use sha2::{Digest, Sha256};
	use std::io::Read;
	let mut file =
		std::fs::File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
	let mut hasher = Sha256::new();
	let mut buf = vec![0u8; 1024 * 1024];
	loop {
		let n = file
			.read(&mut buf)
			.map_err(|e| format!("cannot read {}: {e}", path.display()))?;
		if n == 0 {
			break;
		}
		hasher.update(&buf[..n]);
	}
	Ok(format!("{:x}", hasher.finalize()))
}

/// Update-check verdict for one installed model.
#[derive(Debug, Clone, serde::Serialize)]
pub struct UpdateStatus {
	pub id: &'static str,
	pub file: &'static str,
	/// The local file is byte-identical to the upstream copy.
	pub up_to_date: bool,
}

/// Compares every installed catalog model with huggingface: the tree API
/// supplies the upstream sha256 (`lfs.oid`), the local file is hashed with
/// sha256. Whisper is checked against `main`, GigaAM and Qwen3-ASR against
/// their pinned revisions. Hashing multi-GB files takes seconds, so it runs
/// on a blocking thread.
#[tauri::command]
pub async fn check_model_updates() -> Result<Vec<UpdateStatus>, String> {
	let whisper = fetch_upstream_hashes(HF_TREE).await?;
	let gigaam = fetch_upstream_hashes(GIGAAM_HF_TREE).await?;
	// (id, file, path, upstream sha256; None = no upstream entry)
	let mut jobs: Vec<(&'static str, &'static str, PathBuf, Option<String>)> = Vec::new();
	for &(id, file) in MODEL_CHOICES {
		let path = app_id::models_dir().join(file);
		if path.is_file() {
			jobs.push((id, file, path, whisper.get(file).cloned()));
		}
	}
	for choice in GIGAAM_CHOICES {
		let path = gigaam_dir().join(choice.file);
		if path.is_file() {
			jobs.push((choice.id, choice.file, path, gigaam.get(choice.file).cloned()));
		}
	}
	// One job per file of an installed Qwen3-ASR entry, all under the
	// entry's id; their verdicts are merged below.
	for choice in QWEN_CHOICES {
		let dir = qwen_dir().join(choice.id);
		if !choice.installed() {
			continue;
		}
		let tree = format!(
			"https://huggingface.co/api/models/{}/tree/{}",
			choice.repo, choice.revision
		);
		let upstream = fetch_upstream_hashes(&tree).await?;
		for file in choice.files() {
			jobs.push((choice.id, choice.id, dir.join(file), upstream.get(file).cloned()));
		}
	}
	tauri::async_runtime::spawn_blocking(move || {
		let mut statuses: Vec<UpdateStatus> = Vec::new();
		for (id, file, path, upstream) in jobs {
			let local = sha256_file(&path)?;
			let up_to_date = upstream.as_deref() == Some(local.as_str());
			match statuses.last_mut().filter(|s| s.id == id) {
				Some(status) => status.up_to_date &= up_to_date,
				None => statuses.push(UpdateStatus { id, file, up_to_date }),
			}
		}
		Ok(statuses)
	})
	.await
	.map_err(|e| format!("update check failed: {e}"))?
}
