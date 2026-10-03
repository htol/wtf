//! Qwen3-ASR: the third ASR engine (see DESIGN.md, "Engines"). Multilingual
//! with built-in language detection, on the GPU.
//!
//! The model runs in a `llama-server` child process (llama.cpp, Vulkan
//! build, installed by `models::ensure_llama_runtime`): llama.cpp cannot be
//! linked into this binary next to whisper.cpp, both carry their own ggml.
//! The server listens on a loopback port behind a per-process API key and
//! lives as long as the engine; dropping the engine stops it.
//!
//! A recording goes in as one chat request with a WAV attachment. The model
//! answers `language <Name><asr_text><transcript>`; forcing a language
//! pre-fills that prefix as the start of the assistant turn so only the
//! transcript is generated.

use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use base64::Engine as _;

/// How long the server may take to load the model before it counts as
/// failed.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(120);

/// Upper bound for one transcription request.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(600);

/// Separates the language tag from the transcript in the model's answer.
const SEPARATOR: &str = "<asr_text>";

/// Generation budget: a fixed allowance plus a per-second rate well above
/// real speech, so a decoder stuck in a repetition loop still terminates.
const BASE_TOKENS: usize = 64;
const TOKENS_PER_SEC: usize = 16;

/// Languages the model names in its `language <Name>` tag, with the ISO
/// codes the rest of the app uses.
const LANGUAGES: &[(&str, &str)] = &[
	("zh", "Chinese"),
	("en", "English"),
	("yue", "Cantonese"),
	("ar", "Arabic"),
	("de", "German"),
	("fr", "French"),
	("es", "Spanish"),
	("pt", "Portuguese"),
	("id", "Indonesian"),
	("it", "Italian"),
	("ko", "Korean"),
	("ru", "Russian"),
	("th", "Thai"),
	("vi", "Vietnamese"),
	("ja", "Japanese"),
	("tr", "Turkish"),
	("hi", "Hindi"),
	("ms", "Malay"),
	("nl", "Dutch"),
	("sv", "Swedish"),
	("da", "Danish"),
	("fi", "Finnish"),
	("pl", "Polish"),
	("cs", "Czech"),
	("fil", "Filipino"),
	("fa", "Persian"),
	("el", "Greek"),
	("hu", "Hungarian"),
	("mk", "Macedonian"),
	("ro", "Romanian"),
];

pub struct Qwen {
	server: Child,
	/// `http://127.0.0.1:{port}`.
	url: String,
	api_key: String,
	client: reqwest::Client,
}

impl Qwen {
	/// `model_dir`: the directory holding one catalog entry's files (see
	/// `models::QWEN_CHOICES`). Starts the server and waits until the model
	/// is loaded.
	pub fn new(model_dir: &Path) -> Result<Self, String> {
		let binary = crate::models::llama_server()
			.ok_or("llama.cpp runtime is not installed: re-download the Qwen3-ASR model in settings")?;
		let (model, mmproj) = crate::models::qwen_files(model_dir)
			.ok_or_else(|| format!("{} is not a Qwen3-ASR model dir", model_dir.display()))?;
		// The port is free at this moment; the server binds it right after.
		let port = std::net::TcpListener::bind(("127.0.0.1", 0))
			.and_then(|listener| listener.local_addr())
			.map_err(|e| format!("cannot pick a port for llama-server: {e}"))?
			.port();
		let api_key = random_key()?;
		let server = Command::new(&binary)
			.arg("-m")
			.arg(&model)
			.arg("--mmproj")
			.arg(&mmproj)
			.args(["--gpu-layers", "all", "--host", "127.0.0.1", "--parallel", "1", "--no-webui"])
			.args(["--port", &port.to_string()])
			// In the environment rather than argv: argv is world-readable.
			.env("LLAMA_API_KEY", &api_key)
			.stdin(Stdio::null())
			.spawn()
			.map_err(|e| format!("cannot start {}: {e}", binary.display()))?;
		let mut qwen = Self {
			server,
			url: format!("http://127.0.0.1:{port}"),
			api_key,
			client: reqwest::Client::new(),
		};
		// From here on a failure drops `qwen`, which stops the server.
		qwen.wait_until_ready()?;
		Ok(qwen)
	}

	fn wait_until_ready(&mut self) -> Result<(), String> {
		let started = Instant::now();
		loop {
			if let Some(status) = self.server.try_wait().map_err(|e| e.to_string())? {
				return Err(format!("llama-server exited while loading the model: {status}"));
			}
			let health = self.client.get(format!("{}/health", self.url)).send();
			if tauri::async_runtime::block_on(health).is_ok_and(|r| r.status().is_success()) {
				return Ok(());
			}
			if started.elapsed() > STARTUP_TIMEOUT {
				return Err("llama-server did not become ready in time".into());
			}
			std::thread::sleep(Duration::from_millis(100));
		}
	}

	/// `samples`: 16 kHz mono f32. `language`: ISO code to force, or None for
	/// detection. Returns the transcript and the language code (forced or
	/// detected; "auto" when the model named none).
	pub fn transcribe(
		&mut self,
		samples: &[f32],
		language: Option<&str>,
	) -> Result<(String, String), String> {
		let forced = language.and_then(|code| LANGUAGES.iter().find(|&&(known, _)| known == code));
		if samples.is_empty() {
			return Ok((String::new(), language.unwrap_or("auto").to_string()));
		}
		let audio = base64::engine::general_purpose::STANDARD.encode(wav(samples));
		let mut messages = vec![serde_json::json!({
			"role": "user",
			"content": [{"type": "input_audio", "input_audio": {"data": audio, "format": "wav"}}],
		})];
		if let Some((_, name)) = forced {
			messages.push(serde_json::json!({
				"role": "assistant",
				"content": format!("language {name}{SEPARATOR}"),
			}));
		}
		let secs = samples.len() / crate::audio::SAMPLE_RATE as usize;
		let body = serde_json::json!({
			"messages": messages,
			"temperature": 0,
			"max_tokens": BASE_TOKENS + secs * TOKENS_PER_SEC,
			"cache_prompt": false,
		});
		let request = self
			.client
			.post(format!("{}/v1/chat/completions", self.url))
			.bearer_auth(&self.api_key)
			.header("content-type", "application/json")
			.timeout(REQUEST_TIMEOUT)
			.body(body.to_string());
		let (status, bytes) = tauri::async_runtime::block_on(async {
			let response = request.send().await?;
			Ok::<_, reqwest::Error>((response.status(), response.bytes().await?))
		})
		.map_err(|e| format!("llama-server request failed: {e}"))?;
		if !status.is_success() {
			return Err(format!("llama-server returned {status}: {}", String::from_utf8_lossy(&bytes)));
		}
		let answer: serde_json::Value =
			serde_json::from_slice(&bytes).map_err(|e| format!("bad llama-server response: {e}"))?;
		let content = answer["choices"][0]["message"]["content"]
			.as_str()
			.ok_or("llama-server response has no message content")?;
		match forced {
			// The server echoes the pre-filled prefix in front of the answer.
			Some((code, _)) => {
				let text = content.split_once(SEPARATOR).map_or(content, |(_, text)| text);
				Ok((text.trim().to_string(), code.to_string()))
			}
			None => parse_detected(content),
		}
	}
}

impl Drop for Qwen {
	fn drop(&mut self) {
		let _ = self.server.kill();
		let _ = self.server.wait();
	}
}

/// Splits a detection-mode answer, `language <Name><asr_text>transcript`,
/// into the transcript and the language code ("auto" for a name outside
/// the table, e.g. `None` on non-speech audio).
fn parse_detected(content: &str) -> Result<(String, String), String> {
	// Without the separator the model answered the audio instead of
	// transcribing it.
	let (tag, text) = content
		.split_once(SEPARATOR)
		.ok_or("Qwen3-ASR returned no transcript; pick the language instead of Auto")?;
	let name = tag.trim().strip_prefix("language").unwrap_or(tag).trim();
	let code = LANGUAGES
		.iter()
		.find(|&&(_, known)| known == name)
		.map_or("auto", |&(code, _)| code);
	Ok((text.trim().to_string(), code.to_string()))
}

/// 16 kHz mono f32 samples as a 16-bit PCM WAV file.
fn wav(samples: &[f32]) -> Vec<u8> {
	let data_len = (samples.len() * 2) as u32;
	let rate = crate::audio::SAMPLE_RATE;
	let mut out = Vec::with_capacity(44 + samples.len() * 2);
	out.extend_from_slice(b"RIFF");
	out.extend_from_slice(&(36 + data_len).to_le_bytes());
	out.extend_from_slice(b"WAVEfmt ");
	out.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
	out.extend_from_slice(&1u16.to_le_bytes()); // PCM
	out.extend_from_slice(&1u16.to_le_bytes()); // mono
	out.extend_from_slice(&rate.to_le_bytes());
	out.extend_from_slice(&(rate * 2).to_le_bytes()); // byte rate
	out.extend_from_slice(&2u16.to_le_bytes()); // block align
	out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
	out.extend_from_slice(b"data");
	out.extend_from_slice(&data_len.to_le_bytes());
	for &sample in samples {
		let value = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
		out.extend_from_slice(&value.to_le_bytes());
	}
	out
}

/// A fresh API key for one server process, from the kernel's RNG.
fn random_key() -> Result<String, String> {
	use std::io::Read;
	let mut bytes = [0u8; 16];
	std::fs::File::open("/dev/urandom")
		.and_then(|mut file| file.read_exact(&mut bytes))
		.map_err(|e| format!("cannot read /dev/urandom: {e}"))?;
	Ok(bytes.iter().map(|b| format!("{b:02x}")).collect())
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn detected_answer_splits_into_text_and_code() {
		assert_eq!(
			parse_detected("language Russian<asr_text>Привет, мир.").unwrap(),
			("Привет, мир.".to_string(), "ru".to_string())
		);
		// Non-speech audio: the model names no language and says nothing.
		assert_eq!(
			parse_detected("language None<asr_text>").unwrap(),
			(String::new(), "auto".to_string())
		);
	}

	#[test]
	fn answer_without_separator_is_an_error() {
		assert!(parse_detected("Indeed, I have not to ask for praise.").is_err());
	}

	#[test]
	fn wav_has_a_pcm16_mono_header_and_scaled_samples() {
		let bytes = wav(&[0.0, 1.0, -1.0, 2.0]);
		assert_eq!(bytes.len(), 44 + 8);
		assert_eq!(&bytes[0..4], b"RIFF");
		assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 36 + 8);
		assert_eq!(u32::from_le_bytes(bytes[24..28].try_into().unwrap()), 16_000);
		assert_eq!(u32::from_le_bytes(bytes[40..44].try_into().unwrap()), 8);
		let sample = |i: usize| i16::from_le_bytes(bytes[44 + 2 * i..46 + 2 * i].try_into().unwrap());
		assert_eq!([sample(0), sample(1), sample(2), sample(3)], [0, i16::MAX, -i16::MAX, i16::MAX]);
	}
}
