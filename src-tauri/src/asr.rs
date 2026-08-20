//! Transcription seam over whisper-rs.
//!
//! Runtime GPU backend selection: whisper.cpp registers every backend the
//! binary was built with (cuda, vulkan) and picks a device by index; the index
//! comes from settings (`gpu_device`). Loading a model is expensive, so
//! `Transcriber` (model + whisper state) is created once and cached in app
//! state.

/// A GPU visible to the ASR backend. `index` is the backend's device ordinal
/// that whisper's `gpu_device` setting expects (Vulkan physical-device order
/// or the CUDA ordinal); `pci_bus_id` disambiguates identical cards when the
/// backend can report it.
#[derive(Debug, Clone, serde::Serialize)]
pub struct GpuDevice {
	pub index: i32,
	pub name: String,
	pub pci_bus_id: String,
}

/// Enumerates the GPUs whisper itself will see, in `gpu_device` index order.
/// The Vulkan build asks the Vulkan loader — the same
/// vkEnumeratePhysicalDevices call whisper's backend makes — so indices
/// line up on any vendor driver (RADV, NVIDIA, ...). CUDA-only builds fall
/// back to nvidia-smi (its ordinal == the CUDA index).
#[tauri::command]
pub fn list_gpu_devices() -> Vec<GpuDevice> {
	enumerate_devices()
}

#[cfg(all(feature = "asr-cuda", not(feature = "asr-vulkan")))]
fn enumerate_devices() -> Vec<GpuDevice> {
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

#[cfg(feature = "asr-vulkan")]
fn enumerate_devices() -> Vec<GpuDevice> {
	use ash::vk;
	// The loader is dlopen'd (ash "loaded"): no Vulkan install is needed
	// at build time, and its absence at runtime just means an empty list.
	let entry = match unsafe { ash::Entry::load() } {
		Ok(entry) => entry,
		Err(_) => return Vec::new(),
	};
	let app_info = vk::ApplicationInfo::default().application_name(c"wtf");
	let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
	let Ok(instance) = (unsafe { entry.create_instance(&create_info, None) }) else {
		return Vec::new();
	};
	// Keep `entry` alive until the instance is destroyed: dropping it first
	// would dlclose libvulkan under live function pointers.
	let devices = unsafe { instance.enumerate_physical_devices() }
		.unwrap_or_default()
		.iter()
		.enumerate()
		.map(|(index, &pd)| {
			let props = unsafe { instance.get_physical_device_properties(pd) };
			GpuDevice {
				index: index as i32,
				name: fixed_str(&props.device_name.map(|c| c as u8)),
				pci_bus_id: String::new(),
			}
		})
		.collect();
	unsafe { instance.destroy_instance(None) };
	devices
}

/// Vulkan strings are fixed NUL-terminated buffers; cut at the first NUL,
/// guarding against a missing terminator.
#[cfg(feature = "asr-vulkan")]
fn fixed_str(bytes: &[u8]) -> String {
	bytes
		.split(|&b| b == 0)
		.next()
		.map(|s| String::from_utf8_lossy(s).into_owned())
		.unwrap_or_default()
}

#[cfg(not(any(feature = "asr-vulkan", feature = "asr-cuda")))]
fn enumerate_devices() -> Vec<GpuDevice> {
	Vec::new()
}

#[cfg(all(test, feature = "asr-vulkan"))]
mod tests {
	use super::*;

	#[test]
	fn vulkan_devices_have_names() {
		let devices = enumerate_devices();
		assert!(!devices.is_empty(), "expected a Vulkan ICD on this host");
		for device in &devices {
			assert!(!device.name.is_empty(), "device {} has no name", device.index);
		}
	}
}

#[cfg(feature = "asr")]
pub struct Transcriber {
	/// Reused across transcriptions: `create_state()` allocates the KV cache
	/// and compute buffers (~700 MB on large-v3, ~70 ms) on every call.
	state: whisper_rs::WhisperState,
}

impl Transcriber {
	pub fn new(model_path: &str, gpu_device: i32, use_gpu: bool) -> Result<Self, String> {
		let mut params = whisper_rs::WhisperContextParameters::default();
		params.use_gpu = use_gpu;
		params.gpu_device = gpu_device;
		// Flash attention: ~2x faster on the Vulkan backend (measured on
		// large-v3 q5_0 / RX 9070 XT: 1.72 s -> 0.81 s per 5 s of audio);
		// supported by the CUDA and Vulkan backends.
		params.flash_attn = true;
		let ctx = whisper_rs::WhisperContext::new_with_params(model_path, params)
			.map_err(|e| format!("failed to load model: {e}"))?;
		let state = ctx
			.create_state()
			.map_err(|e| format!("failed to create whisper state: {e}"))?;
		Ok(Self { state })
	}

	/// `samples`: 16 kHz mono f32. `language`: language code, "auto", or None.
	/// `initial_prompt`: optional decoder conditioning text (style/vocabulary
	/// bias, e.g. code-switching examples); no instructions are understood.
	/// Returns the transcript and the effective language code (forced or
	/// detected).
	pub fn transcribe(
		&mut self,
		samples: &[f32],
		language: Option<&str>,
		initial_prompt: Option<&str>,
	) -> Result<(String, String), String> {
		let mut params =
			whisper_rs::FullParams::new(whisper_rs::SamplingStrategy::Greedy { best_of: 1 });
		let language = match language {
			None | Some("auto") => None,
			some => some,
		};
		params.set_language(language);
		if let Some(prompt) = initial_prompt {
			params.set_initial_prompt(prompt);
		}
		let mut state = &mut self.state;
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
		let lang = match language {
			Some(code) => code.to_string(),
			None => lang_code(state.full_lang_id_from_state()),
		};
		Ok((out.trim().to_string(), lang))
	}
}

/// Whisper language ids (whisper.cpp `g_lang` ordering) -> ISO codes.
const LANG_CODES: &[&str] = &[
	"en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar",
	"sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu",
	"ta", "no", "th", "ur", "hr", "bg", "lt", "la",
];

/// Maps a whisper language id to its ISO code ("auto" when detection
/// did not run, `id:N` for ids outside the table above).
fn lang_code(id: i32) -> String {
	match id {
		-1 => "auto".into(),
		i => LANG_CODES
			.get(i as usize)
			.map(|code| code.to_string())
			.unwrap_or_else(|| format!("id:{i}")),
	}
}

#[cfg(not(feature = "asr"))]
pub struct Transcriber;

#[cfg(not(feature = "asr"))]
impl Transcriber {
	pub fn new(_model_path: &str, _gpu_device: i32, _use_gpu: bool) -> Result<Self, String> {
		Err("built without the `asr` feature".into())
	}

	pub fn transcribe(
		&mut self,
		_samples: &[f32],
		_language: Option<&str>,
		_initial_prompt: Option<&str>,
	) -> Result<(String, String), String> {
		Err("built without the `asr` feature".into())
	}
}
