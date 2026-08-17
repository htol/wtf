//! Single source of truth for the application identity and its on-disk
//! locations. When the working name `wtf` is replaced, change it here once and
//! migrate the data directories (see DESIGN.md, "App identity").

pub const APP_ID: &str = "wtf";

pub fn config_dir() -> std::path::PathBuf {
	dirs::config_dir()
		.unwrap_or_else(|| std::path::PathBuf::from("."))
		.join(APP_ID)
}

pub fn data_dir() -> std::path::PathBuf {
	dirs::data_dir()
		.unwrap_or_else(|| std::path::PathBuf::from("."))
		.join(APP_ID)
}

pub fn models_dir() -> std::path::PathBuf {
	data_dir().join("models")
}
