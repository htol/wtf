//! Transcription seam over whisper-rs.
//!
//! Runtime GPU backend selection: whisper.cpp registers every backend the
//! binary was built with (cuda, vulkan) and picks a device by index; the index
//! comes from settings (`gpu_device`). Loading a model is expensive, so
//! `Transcriber` will be created once and cached in app state.

/// A GPU visible to the ASR backend. `index` is the CUDA ordinal that
/// whisper's `gpu_device` setting expects; `pci_bus_id` is the bus address.
#[derive(Debug, Clone, serde::Serialize)]
pub struct GpuDevice {
	pub index: i32,
	pub name: String,
	pub pci_bus_id: String,
}

/// Enumerates NVIDIA GPUs (nvidia-smi ordinal == CUDA device index).
/// Returns an empty list when nvidia-smi is unavailable.
#[tauri::command]
pub fn list_gpu_devices() -> Vec<GpuDevice> {
	let output = std::process::Command::new("nvidia-smi")
		.args(["--query-gpu=index,name,pci.bus_id", "--format=csv,noheader"])
		.output();
	let Ok(output) = output else {
		return Vec::new();
	};
	if !output.status.success() {
		return Vec::new();
	}
	String::from_utf8_lossy(&output.stdout)
		.lines()
		.filter_map(|line| {
			let mut parts = line.split(", ");
			let index = parts.next()?.trim().parse().ok()?;
			let name = parts.next()?.trim().to_string();
			let pci_bus_id = parts.next().unwrap_or("").trim().to_string();
			Some(GpuDevice {
				index,
			name,
			pci_bus_id,
			})
		})
		.collect()
}

#[cfg(feature = "asr")]
pub struct Transcriber {
	#[allow(dead_code)] // placeholder until model loading is implemented
	ctx: whisper_rs::WhisperContext,
}

impl Transcriber {
	pub fn new(model_path: &str, gpu_device: i32, use_gpu: bool) -> Result<Self, String> {
		let mut params = whisper_rs::WhisperContextParameters::default();
		params.use_gpu = use_gpu;
		params.gpu_device = gpu_device;
		let ctx = whisper_rs::WhisperContext::new_with_params(model_path, params)
			.map_err(|e| format!("failed to load model: {e}"))?;
		Ok(Self { ctx })
	}

	/// `samples`: 16 kHz mono f32. `language`: language code, "auto", or None.
	pub fn transcribe(&self, samples: &[f32], language: Option<&str>) -> Result<String, String> {
		let mut params =
			whisper_rs::FullParams::new(whisper_rs::SamplingStrategy::Greedy { best_of: 1 });
		let language = match language {
			None | Some("auto") => None,
			some => some,
		};
		params.set_language(language);
		let mut state = self.ctx.create_state().map_err(|e| e.to_string())?;
		state
			.full(params, samples)
			.map_err(|e| format!("transcription failed: {e}"))?;
		let n = state.full_n_segments();
		let mut out = String::new();
		for i in 0..n {
			if let Some(segment) = state.get_segment(i) {
				out.push_str(segment.to_str().map_err(|e| e.to_string())?);
			}
		}
		Ok(out.trim().to_string())
	}
}

#[cfg(not(feature = "asr"))]
pub struct Transcriber;

#[cfg(not(feature = "asr"))]
impl Transcriber {
	pub fn new(_model_path: &str, _gpu_device: i32, _use_gpu: bool) -> Result<Self, String> {
		Err("built without the `asr` feature".into())
	}

	pub fn transcribe(&self, _samples: &[f32], _language: Option<&str>) -> Result<String, String> {
		Err("built without the `asr` feature".into())
	}
}
