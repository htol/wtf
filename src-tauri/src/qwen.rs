//! Qwen3-ASR: the third ASR engine (see DESIGN.md, "Engines"). Multilingual
//! with built-in language detection, CPU only, via ONNX Runtime.
//!
//! Model files are the int4 ONNX export from HF `andrewleech/qwen3-asr-*-onnx`
//! (downloaded by `models::download_qwen_model`). Three `ort` sessions:
//! - encoder: log-mel `[1, 128, T]` -> audio features `[1, N, H]`;
//! - decoder_init: the whole chat prompt in one pass (prefill), with the
//!   audio features scattered over the `<|audio_pad|>` positions; returns
//!   logits and the KV cache;
//! - decoder_step: one token per call, carrying the KV cache.
//!
//! The log-mel front end is Whisper's (128 bins, n_fft 400, hop 160, Slaney
//! scale) and is computed here. Decoding is greedy. The model answers
//! `language <Name><asr_text><transcript>`; forcing a language pre-fills
//! that prefix so only the transcript is generated.
//!
//! Long recordings are cut at silences (`vad`) into pieces of up to
//! `MAX_PIECE_SECS`: decoding cost grows faster than length.

use std::collections::HashMap;
use std::fs::File;
use std::ops::Range;
use std::os::unix::fs::FileExt;
use std::path::Path;

use ort::session::Session;
use ort::value::Tensor;
use rustfft::num_complex::Complex;

const N_FFT: usize = 400;
const HOP_LENGTH: usize = 160;
const N_MELS: usize = 128;

/// Mel frames per encoder convolution window, and the audio tokens one full
/// window yields (native Qwen3-ASR encoder geometry).
const ENCODER_WINDOW_FRAMES: usize = 100;
const ENCODER_WINDOW_TOKENS: usize = 13;

/// Generation budget: a fixed allowance plus a per-second rate well above
/// real speech, so a decoder stuck in a repetition loop still terminates.
const BASE_TOKENS: usize = 64;
const TOKENS_PER_SEC: usize = 16;
const MAX_TOKENS: usize = 4096;

/// Longest piece decoded in one pass; shorter recordings skip VAD. Measured
/// with 0.6B int4: 71 s decodes in 41 s whole and in 20 s in pieces.
const MAX_PIECE_SECS: f32 = 30.0;

/// GPT-2 byte-level BPE spells a space as U+0120 and a newline as U+010A.
const BPE_SPACE: char = 'Ġ';
const BPE_NEWLINE: &str = "Ċ";

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

/// The slice of `config.json` the engine needs.
#[derive(serde::Deserialize)]
struct Config {
	decoder: DecoderConfig,
	special_tokens: SpecialTokens,
	embed_tokens_dtype: String,
}

#[derive(serde::Deserialize)]
struct DecoderConfig {
	hidden_size: usize,
	vocab_size: usize,
}

#[derive(serde::Deserialize)]
struct SpecialTokens {
	eos_token_ids: Vec<i64>,
	im_start_token_id: i64,
	im_end_token_id: i64,
	audio_start_token_id: i64,
	audio_end_token_id: i64,
	audio_pad_token_id: i64,
	asr_text_token_id: i64,
}

pub struct Qwen {
	/// All sessions take `&mut self` to run; the engine sits behind the
	/// pipeline's mutex anyway.
	encoder: Session,
	decoder_init: Session,
	decoder_step: Session,
	/// `embed_tokens.bin`: `[vocab, hidden]` fp16, row-major. decoder_step
	/// takes the embedding of the previous token as input; rows are read on
	/// demand (page cache) instead of holding the table in memory.
	embed: File,
	hidden: usize,
	vocab_size: usize,
	special: SpecialTokens,
	/// Raw bytes per token id (byte-level BPE, already unescaped).
	vocab: Vec<Vec<u8>>,
	/// Ids of the plain-text prompt tokens.
	system_id: i64,
	user_id: i64,
	assistant_id: i64,
	newline_id: i64,
	language_id: i64,
	/// ISO code -> id of the ` <Name>` token, for languages whose name is a
	/// single token (the rest cannot be forced and fall back to detection).
	language_ids: HashMap<&'static str, i64>,
	fft: std::sync::Arc<dyn rustfft::Fft<f64>>,
	mel_filters: Vec<MelFilter>,
	vad: crate::vad::Vad,
}

/// One triangular mel filter: weights for FFT bins `start..start + len`.
struct MelFilter {
	start: usize,
	weights: Vec<f64>,
}

impl Qwen {
	/// `model_dir`: the directory holding one catalog entry's files (see
	/// `models::QWEN_FILES`).
	pub fn new(model_dir: &Path) -> Result<Self, String> {
		let read = |file: &str| {
			let path = model_dir.join(file);
			std::fs::read_to_string(&path).map_err(|e| format!("cannot read {}: {e}", path.display()))
		};
		let config: Config = serde_json::from_str(&read(crate::models::QWEN_CONFIG_FILE)?)
			.map_err(|e| format!("bad qwen config: {e}"))?;
		if config.embed_tokens_dtype != "float16" {
			return Err(format!("unsupported embed_tokens dtype {:?}", config.embed_tokens_dtype));
		}
		let tokenizer: serde_json::Value = serde_json::from_str(&read("tokenizer.json")?)
			.map_err(|e| format!("bad qwen tokenizer: {e}"))?;
		let tokens = tokenizer["model"]["vocab"]
			.as_object()
			.ok_or("qwen tokenizer has no vocab")?;
		let id_of = |token: &str| tokens.get(token).and_then(|id| id.as_i64());
		let require = |token: &str| id_of(token).ok_or_else(|| format!("qwen vocab lacks {token:?}"));
		let language_ids = LANGUAGES
			.iter()
			.filter_map(|&(code, name)| Some((code, id_of(&format!("{BPE_SPACE}{name}"))?)))
			.collect();
		let mut vocab: Vec<Vec<u8>> = Vec::new();
		for (token, id) in tokens {
			let Some(id) = id.as_u64() else { continue };
			let id = id as usize;
			if vocab.len() <= id {
				vocab.resize(id + 1, Vec::new());
			}
			vocab[id] = token.chars().filter_map(bpe_byte).collect();
		}

		let session = |file: &str| {
			let path = model_dir.join(file);
			ort::session::Session::builder()
				.map_err(|e| format!("cannot create session builder: {e}"))?
				.commit_from_file(&path)
				.map_err(|e| format!("failed to load model {}: {e}", path.display()))
		};
		let embed_path = model_dir.join("embed_tokens.bin");
		let embed =
			File::open(&embed_path).map_err(|e| format!("cannot open {}: {e}", embed_path.display()))?;
		Ok(Self {
			encoder: session("encoder.int4.onnx")?,
			decoder_init: session("decoder_init.int4.onnx")?,
			decoder_step: session("decoder_step.int4.onnx")?,
			embed,
			hidden: config.decoder.hidden_size,
			vocab_size: config.decoder.vocab_size,
			system_id: require("system")?,
			user_id: require("user")?,
			assistant_id: require("assistant")?,
			newline_id: require(BPE_NEWLINE)?,
			language_id: require("language")?,
			language_ids,
			special: config.special_tokens,
			vocab,
			fft: rustfft::FftPlanner::new().plan_fft_forward(N_FFT),
			mel_filters: mel_filters(),
			vad: crate::vad::Vad::new()?,
		})
	}

	/// `samples`: 16 kHz mono f32. `language`: ISO code to force, or None for
	/// detection. Returns the transcript and the language code (forced or
	/// detected; "auto" when the model named none).
	pub fn transcribe(
		&mut self,
		samples: &[f32],
		language: Option<&str>,
	) -> Result<(String, String), String> {
		let max = (MAX_PIECE_SECS * crate::audio::SAMPLE_RATE as f32) as usize;
		let islands = self.vad.speech_segments(samples, MAX_PIECE_SECS, MAX_PIECE_SECS);
		// The language detected on the first piece is forced on the rest.
		let mut detected: Option<String> = None;
		let mut parts: Vec<String> = Vec::new();
		for range in pack(&islands, max) {
			let (text, code) =
				self.transcribe_piece(&samples[range], language.or(detected.as_deref()))?;
			if detected.is_none() && code != "auto" {
				detected = Some(code);
			}
			if !text.is_empty() {
				parts.push(text);
			}
		}
		let code = detected.unwrap_or_else(|| language.unwrap_or("auto").to_string());
		Ok((parts.join(" "), code))
	}

	/// One piece through mel -> encoder -> decoder.
	fn transcribe_piece(
		&mut self,
		samples: &[f32],
		language: Option<&str>,
	) -> Result<(String, String), String> {
		let forced = language.and_then(|code| Some((code, *self.language_ids.get(code)?)));
		let (mel, frames) = log_mel(samples, self.fft.as_ref(), &self.mel_filters);
		if frames == 0 {
			return Ok((String::new(), language.unwrap_or("auto").to_string()));
		}
		let mel = Tensor::from_array((vec![1usize, N_MELS, frames], mel)).map_err(|e| e.to_string())?;
		let encoded = self
			.encoder
			.run(ort::inputs!["mel" => mel])
			.map_err(|e| format!("encoder failed: {e}"))?;
		let (shape, features) = encoded["audio_features"]
			.try_extract_tensor::<f32>()
			.map_err(|e| e.to_string())?;
		let audio_tokens = shape[1] as usize;
		if audio_tokens != audio_token_count(frames) {
			return Err(format!(
				"encoder produced {audio_tokens} audio tokens for {frames} mel frames, expected {}",
				audio_token_count(frames)
			));
		}
		let features = Tensor::from_array((vec![1usize, audio_tokens, self.hidden], features.to_vec()))
			.map_err(|e| e.to_string())?;
		drop(encoded);

		let prompt = self.prompt(audio_tokens, forced.map(|(_, id)| id));
		let audio_offset = prompt
			.iter()
			.position(|&id| id == self.special.audio_pad_token_id)
			.expect("prompt has audio pads") as i64;
		let len = prompt.len();
		let positions: Vec<i64> = (0..len as i64).collect();
		let input_ids = Tensor::from_array((vec![1usize, len], prompt)).map_err(|e| e.to_string())?;
		let position_ids =
			Tensor::from_array((vec![1usize, len], positions)).map_err(|e| e.to_string())?;
		let audio_offset =
			Tensor::from_array((vec![1usize], vec![audio_offset])).map_err(|e| e.to_string())?;
		let mut out = self
			.decoder_init
			.run(ort::inputs![
				"input_ids" => input_ids,
				"position_ids" => position_ids,
				"audio_features" => features,
				"audio_offset" => audio_offset
			])
			.map_err(|e| format!("decoder prefill failed: {e}"))?;
		let mut token = last_argmax(&out["logits"], self.vocab_size)?;
		let mut keys = out.remove("present_keys").ok_or("decoder_init returned no present_keys")?;
		let mut values =
			out.remove("present_values").ok_or("decoder_init returned no present_values")?;
		drop(out);

		let secs = samples.len() / crate::audio::SAMPLE_RATE as usize;
		let budget = (BASE_TOKENS + secs * TOKENS_PER_SEC).min(MAX_TOKENS);
		let mut generated: Vec<i64> = Vec::new();
		let mut position = len as i64;
		let mut row = vec![0u8; self.hidden * 2];
		while !self.special.eos_token_ids.contains(&token) && generated.len() < budget {
			generated.push(token);
			let row_offset = token as u64 * row.len() as u64;
			self.embed
				.read_exact_at(&mut row, row_offset)
				.map_err(|e| format!("cannot read embedding of token {token}: {e}"))?;
			let embedding: Vec<f32> = row
				.chunks_exact(2)
				.map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
				.collect();
			let input_embeds = Tensor::from_array((vec![1usize, 1, self.hidden], embedding))
				.map_err(|e| e.to_string())?;
			let position_ids =
				Tensor::from_array((vec![1usize, 1], vec![position])).map_err(|e| e.to_string())?;
			let mut out = self
				.decoder_step
				.run(ort::inputs![
					"input_embeds" => input_embeds,
					"position_ids" => position_ids,
					"past_keys" => keys,
					"past_values" => values
				])
				.map_err(|e| format!("decoder step failed: {e}"))?;
			token = last_argmax(&out["logits"], self.vocab_size)?;
			keys = out.remove("present_keys").ok_or("decoder_step returned no present_keys")?;
			values = out.remove("present_values").ok_or("decoder_step returned no present_values")?;
			position += 1;
		}

		if let Some((code, _)) = forced {
			return Ok((self.decode(&generated), code.to_string()));
		}
		// Detection: `language <Name>` `<asr_text>` transcript. Without the
		// separator the model answered the audio instead of transcribing it
		// (seen with 0.6B int4: an English paraphrase of Russian speech).
		let Some(split) = generated.iter().position(|&id| id == self.special.asr_text_token_id) else {
			return Err("Qwen3-ASR returned no transcript; pick the language instead of Auto".into());
		};
		let tag = self.decode(&generated[..split]);
		let name = tag.strip_prefix("language").unwrap_or(&tag).trim();
		let code = LANGUAGES
			.iter()
			.find(|&&(_, known)| known == name)
			.map(|&(code, _)| code)
			.unwrap_or("auto");
		Ok((self.decode(&generated[split + 1..]), code.to_string()))
	}

	/// The chat prompt: empty system turn, user turn holding the audio
	/// placeholders, and the opened assistant turn. With a forced language
	/// the assistant's answer is pre-filled up to `<asr_text>`.
	fn prompt(&self, audio_tokens: usize, forced_language: Option<i64>) -> Vec<i64> {
		let s = &self.special;
		let mut ids = vec![
			s.im_start_token_id,
			self.system_id,
			self.newline_id,
			s.im_end_token_id,
			self.newline_id,
			s.im_start_token_id,
			self.user_id,
			self.newline_id,
			s.audio_start_token_id,
		];
		ids.extend(std::iter::repeat_n(s.audio_pad_token_id, audio_tokens));
		ids.extend([
			s.audio_end_token_id,
			s.im_end_token_id,
			self.newline_id,
			s.im_start_token_id,
			self.assistant_id,
			self.newline_id,
		]);
		if let Some(name_id) = forced_language {
			ids.extend([self.language_id, name_id, s.asr_text_token_id]);
		}
		ids
	}

	/// Token ids -> text. Ids outside the BPE vocab (special and added
	/// tokens) carry no text and are skipped.
	fn decode(&self, ids: &[i64]) -> String {
		let bytes: Vec<u8> = ids
			.iter()
			.filter_map(|&id| self.vocab.get(id as usize))
			.flatten()
			.copied()
			.collect();
		String::from_utf8_lossy(&bytes).trim().to_string()
	}
}

/// Joins consecutive speech islands into pieces spanning at most `max`
/// samples (silence between joined islands included): every piece costs a
/// decoder prefill and loses the context of its neighbours.
fn pack(islands: &[Range<usize>], max: usize) -> Vec<Range<usize>> {
	let mut pieces: Vec<Range<usize>> = Vec::new();
	for island in islands {
		match pieces.last_mut() {
			Some(piece) if island.end - piece.start <= max => piece.end = island.end,
			_ => pieces.push(island.clone()),
		}
	}
	pieces
}

/// Argmax over the last position of a `[1, T, vocab]` logits tensor.
fn last_argmax(logits: &ort::value::DynValue, vocab_size: usize) -> Result<i64, String> {
	let (_, data) = logits.try_extract_tensor::<f32>().map_err(|e| e.to_string())?;
	if data.len() < vocab_size {
		return Err(format!("logits hold {} values, expected at least {vocab_size}", data.len()));
	}
	let last = &data[data.len() - vocab_size..];
	let best = last
		.iter()
		.enumerate()
		.max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
		.map(|(i, _)| i)
		.unwrap_or(0);
	Ok(best as i64)
}

/// Audio tokens the encoder emits for a mel frame count: 13 per full
/// 100-frame window, the remainder through three stride-2 convolutions.
fn audio_token_count(frames: usize) -> usize {
	let rest = frames % ENCODER_WINDOW_FRAMES;
	let rest = rest.div_ceil(2).div_ceil(2).div_ceil(2);
	rest + frames / ENCODER_WINDOW_FRAMES * ENCODER_WINDOW_TOKENS
}

/// Inverse of the GPT-2 byte-level BPE alphabet: printable bytes are
/// spelled as themselves, the other 68 as U+0100.. in byte order.
fn bpe_byte(c: char) -> Option<u8> {
	match c as u32 {
		cp @ (0x21..=0x7e | 0xa1..=0xac | 0xae..=0xff) => Some(cp as u8),
		cp @ 0x100..=0x120 => Some((cp - 0x100) as u8),
		cp @ 0x121..=0x142 => Some((cp - 0x121 + 0x7f) as u8),
		0x143 => Some(0xad),
		_ => None,
	}
}

/// IEEE 754 half precision -> f32.
fn f16_to_f32(h: u16) -> f32 {
	let sign = ((h >> 15) as u32) << 31;
	let exp = ((h >> 10) & 0x1f) as u32;
	let frac = (h & 0x3ff) as u32;
	match exp {
		0 => {
			// Zero and subnormals: frac * 2^-24.
			let magnitude = frac as f32 * (1.0 / 16_777_216.0);
			f32::from_bits(sign | magnitude.to_bits())
		}
		0x1f => f32::from_bits(sign | 0x7f80_0000 | (frac << 13)),
		_ => f32::from_bits(sign | ((exp + 112) << 23) | (frac << 13)),
	}
}

/// Whisper-style log-mel spectrogram, `[N_MELS, frames]` row-major, plus
/// the frame count. Mirrors `WhisperFeatureExtractor`: centered STFT with
/// reflect padding, periodic Hann window, power spectrum, Slaney mel
/// filters, log10 clamped to 8 below the peak, scaled by (x + 4) / 4, last
/// STFT frame dropped.
fn log_mel(samples: &[f32], fft: &dyn rustfft::Fft<f64>, filters: &[MelFilter]) -> (Vec<f32>, usize) {
	let half = N_FFT / 2;
	// Reflect padding needs more than half a window of audio.
	if samples.len() <= half {
		return (Vec::new(), 0);
	}
	let frames = samples.len() / HOP_LENGTH;
	let window: Vec<f64> = (0..N_FFT)
		.map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / N_FFT as f64).cos()))
		.collect();
	let last = samples.len() as isize - 1;
	let mut buffer = vec![Complex::new(0.0f64, 0.0); N_FFT];
	let mut power = vec![0.0f64; half + 1];
	let mut mel = vec![0.0f32; N_MELS * frames];
	let mut peak = f32::NEG_INFINITY;
	for frame in 0..frames {
		let start = (frame * HOP_LENGTH) as isize - half as isize;
		for (i, slot) in buffer.iter_mut().enumerate() {
			let mut index = start + i as isize;
			if index < 0 {
				index = -index;
			} else if index > last {
				index = 2 * last - index;
			}
			*slot = Complex::new(samples[index as usize] as f64 * window[i], 0.0);
		}
		fft.process(&mut buffer);
		for (bin, value) in power.iter_mut().enumerate() {
			*value = buffer[bin].norm_sqr();
		}
		for (m, filter) in filters.iter().enumerate() {
			let energy: f64 = filter
				.weights
				.iter()
				.zip(&power[filter.start..])
				.map(|(w, p)| w * p)
				.sum();
			let value = energy.max(1e-10).log10() as f32;
			peak = peak.max(value);
			mel[m * frames + frame] = value;
		}
	}
	for value in mel.iter_mut() {
		*value = (value.max(peak - 8.0) + 4.0) / 4.0;
	}
	(mel, frames)
}

/// Slaney-scale, area-normalized triangular filters over 0-8 kHz (what
/// `librosa.filters.mel(norm="slaney")` yields for these parameters).
fn mel_filters() -> Vec<MelFilter> {
	// Slaney mel scale: linear below 1 kHz, logarithmic above.
	const LINEAR_HZ_PER_MEL: f64 = 200.0 / 3.0;
	const LOG_START_HZ: f64 = 1000.0;
	let log_start_mel = LOG_START_HZ / LINEAR_HZ_PER_MEL;
	let log_step = 6.4f64.ln() / 27.0;
	let hz_to_mel = |hz: f64| {
		if hz < LOG_START_HZ {
			hz / LINEAR_HZ_PER_MEL
		} else {
			log_start_mel + (hz / LOG_START_HZ).ln() / log_step
		}
	};
	let mel_to_hz = |mel: f64| {
		if mel < log_start_mel {
			mel * LINEAR_HZ_PER_MEL
		} else {
			LOG_START_HZ * ((mel - log_start_mel) * log_step).exp()
		}
	};
	let nyquist = crate::audio::SAMPLE_RATE as f64 / 2.0;
	let mel_max = hz_to_mel(nyquist);
	let edges: Vec<f64> = (0..N_MELS + 2)
		.map(|i| mel_to_hz(mel_max * i as f64 / (N_MELS + 1) as f64))
		.collect();
	let bin_hz = crate::audio::SAMPLE_RATE as f64 / N_FFT as f64;
	(0..N_MELS)
		.map(|m| {
			let (left, center, right) = (edges[m], edges[m + 1], edges[m + 2]);
			let norm = 2.0 / (right - left);
			let dense: Vec<f64> = (0..=N_FFT / 2)
				.map(|bin| {
					let hz = bin as f64 * bin_hz;
					let rising = (hz - left) / (center - left);
					let falling = (right - hz) / (right - center);
					norm * rising.min(falling).max(0.0)
				})
				.collect();
			let start = dense.iter().position(|&w| w > 0.0).unwrap_or(0);
			let end = dense.iter().rposition(|&w| w > 0.0).map_or(start, |i| i + 1);
			MelFilter {
				start,
				weights: dense[start..end].to_vec(),
			}
		})
		.collect()
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn audio_token_count_matches_encoder_geometry() {
		assert_eq!(audio_token_count(0), 0);
		assert_eq!(audio_token_count(1), 1);
		assert_eq!(audio_token_count(99), 13);
		assert_eq!(audio_token_count(100), 13);
		assert_eq!(audio_token_count(200), 26);
		assert_eq!(audio_token_count(997), 130);
	}

	#[test]
	fn pack_joins_islands_up_to_the_limit() {
		// 0..40 fits; adding 50..90 stays within 100; 95..130 would span 130.
		assert_eq!(pack(&[0..40, 50..90, 95..130, 140..150], 100), vec![0..90, 95..150]);
		assert_eq!(pack(&[0..10], 100), vec![0..10]);
		assert!(pack(&[], 100).is_empty());
	}

	#[test]
	fn bpe_alphabet_covers_every_byte_once() {
		let mut seen = [false; 256];
		for cp in 0x21..=0x143u32 {
			if let Some(byte) = bpe_byte(char::from_u32(cp).unwrap()) {
				assert!(!seen[byte as usize], "byte {byte:#x} mapped twice");
				seen[byte as usize] = true;
			}
		}
		assert!(seen.iter().all(|&s| s));
		assert_eq!(bpe_byte(BPE_SPACE), Some(b' '));
		assert_eq!(bpe_byte('Ċ'), Some(b'\n'));
		assert_eq!(bpe_byte('A'), Some(b'A'));
	}

	#[test]
	fn f16_conversion_handles_all_classes() {
		assert_eq!(f16_to_f32(0x3c00), 1.0);
		assert_eq!(f16_to_f32(0xbc00), -1.0);
		assert_eq!(f16_to_f32(0x3800), 0.5);
		assert_eq!(f16_to_f32(0x7bff), 65504.0);
		assert_eq!(f16_to_f32(0x0001), 2.0f32.powi(-24));
		assert!(f16_to_f32(0x8000).is_sign_negative());
		assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
		assert!(f16_to_f32(0x7e00).is_nan());
	}

	#[test]
	fn mel_filters_are_slaney_normalized_triangles() {
		let filters = mel_filters();
		assert_eq!(filters.len(), N_MELS);
		for filter in &filters {
			assert!(!filter.weights.is_empty());
			assert!(filter.start + filter.weights.len() <= N_FFT / 2 + 1);
			assert!(filter.weights.iter().all(|&w| w >= 0.0));
		}
		// Slaney normalization: each triangle has unit area over Hz. The top
		// band spans enough 40 Hz bins for the sampled area to show it.
		let area: f64 = filters[N_MELS - 1].weights.iter().sum::<f64>() * 40.0;
		assert!((area - 1.0).abs() < 0.05, "top band area {area}");
	}

	#[test]
	fn log_mel_shape_and_range() {
		let fft = rustfft::FftPlanner::new().plan_fft_forward(N_FFT);
		let filters = mel_filters();
		// Too short for reflect padding.
		assert_eq!(log_mel(&[0.0; 200], fft.as_ref(), &filters).1, 0);
		// 1 s of a 440 Hz tone: 100 frames, energy concentrated in low bins.
		let tone: Vec<f32> = (0..16_000)
			.map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16_000.0).sin())
			.collect();
		let (mel, frames) = log_mel(&tone, fft.as_ref(), &filters);
		assert_eq!(frames, 100);
		assert_eq!(mel.len(), N_MELS * frames);
		let band_mean = |m: usize| mel[m * frames..(m + 1) * frames].iter().sum::<f32>() / frames as f32;
		let loudest = (0..N_MELS)
			.max_by(|&a, &b| band_mean(a).partial_cmp(&band_mean(b)).unwrap())
			.unwrap();
		// 440 Hz is 6.6 mel (linear region); band centers are spaced
		// 45.25 / 129 mel apart, which puts it at band 17-18.
		assert!((16..=19).contains(&loudest), "loudest band {loudest}");
		let max = mel.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
		let min = mel.iter().cloned().fold(f32::INFINITY, f32::min);
		assert!(max - min <= 2.0 + 1e-6, "dynamic range is clamped to 8 decades, scaled by 1/4");
	}
}
