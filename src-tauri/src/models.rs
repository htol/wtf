//! Model management: download into the models dir on first run, or use a
//! manual path override from settings.
//!
//! Default recommendation: ggml-large-v3-turbo q5_0 from
//! https://huggingface.co/ggerganov/whisper.cpp
//! (multilingual models, see DESIGN.md "Models").

use crate::app_id;

/// Candidate models for the first-run picker.
pub const MODEL_CHOICES: &[(&str, &str)] = &[
	// (id, huggingface ggml file name)
	("tiny", "ggml-tiny.bin"),
	("base", "ggml-base.bin"),
	("small", "ggml-small.bin"),
	("medium", "ggml-medium.bin"),
	("large-v3-turbo-q5_0", "ggml-large-v3-turbo-q5_0.bin"),
];

/// Resolves the model file to use: manual override from settings, else the
/// most recent download in the models dir, else None (first-run picker).
pub fn resolve(manual_path: Option<&str>) -> Option<std::path::PathBuf> {
	if let Some(p) = manual_path {
		return Some(std::path::PathBuf::from(p));
	}
	let dir = app_id::models_dir();
	let latest = std::fs::read_dir(&dir)
		.ok()?
		.filter_map(|e| e.ok())
		.filter(|e| e.path().extension().is_some_and(|x| x == "bin"))
		.max_by_key(|e| e.metadata().ok().and_then(|m| m.modified().ok()));
	latest.map(|e| e.path())
}

// TODO: download(model_id) -> stream the file from huggingface into models_dir
// with progress events to the frontend; then first-run picker UI.
