//! GigaAM v3 e2e-CTC: the second ASR engine next to whisper (see DESIGN.md,
//! "Engines"). Russian speech, CPU only, via ONNX Runtime.
//!
//! Three `ort` sessions per engine instance:
//! - log-mel preprocessor graph, vendored from onnx-asr (MIT) and run as-is —
//!   zero parity risk with the reference Python toolchain;
//! - the acoustic model (`v3_e2e_ctc[.int8].onnx`, downloaded to the models
//!   dir by `models::download_gigaam_model`);
//! - Silero VAD (`vad`), used to split long recordings: the model's segment
//!   limit is ~25 s, so anything longer is cut into speech islands first.
//!
//! Decoding is CTC greedy: argmax per 40 ms frame, drop blanks, collapse
//! repeats, map ids to tokens. Verified byte-identical to `onnx_asr.recognize`
//! on the official GigaAM sample.

use std::path::Path;

use ort::session::Session;
use ort::value::Tensor;

/// Torchaudio-style log-mel graph: `waveforms [B, L]` f32 @16 kHz +
/// `waveforms_lens [B]` i64 -> `features [B, 64, T]` + `features_lens [B]`.
/// Extracted from the onnx-asr wheel (MIT), revision pinned by the file
/// itself; see DESIGN.md "Engines".
const MEL_GRAPH: &[u8] = include_bytes!("../assets/gigaam_mel.onnx");

/// Recordings shorter than this are transcribed whole, without VAD.
const VAD_MIN_DURATION_SECS: f32 = 20.0;

/// Longest single piece handed to the model. The graph's practical segment
/// limit is ~25 s; stay below it.
const MAX_SEGMENT_SECS: f32 = 20.0;

/// CTC blank id (last vocab entry, `<blk>`).
const BLANK_ID: usize = 256;

/// Sentence-piece space marker (U+2581).
const SPACE_TOKEN: char = '▁';

pub struct GigaAm {
	/// All sessions take `&mut self` to run; the engine sits behind the
	/// pipeline's mutex anyway.
	mel: Session,
	model: Session,
	vad: crate::vad::Vad,
	/// Token per id, `▁` already replaced with a space (as onnx-asr does).
	vocab: Vec<String>,
}

impl GigaAm {
	/// `model_path`: the downloaded `.onnx` file; the vocab file is expected
	/// next to it (both come from `models::download_gigaam_model`).
	pub fn new(model_path: &Path) -> Result<Self, String> {
		let builder = || {
			ort::session::Session::builder()
				.map_err(|e| format!("cannot create session builder: {e}"))
		};
		let mel = builder()?
			.commit_from_memory(MEL_GRAPH)
			.map_err(|e| format!("cannot load mel preprocessor: {e}"))?;
		let vad = crate::vad::Vad::new()?;
		let model = builder()?
			.commit_from_file(model_path)
			.map_err(|e| format!("failed to load model {}: {e}", model_path.display()))?;
		let vocab_path = model_path.with_file_name(crate::models::GIGAAM_VOCAB_FILE);
		let vocab = load_vocab(&vocab_path)?;
		Ok(Self { mel, model, vad, vocab })
	}

	/// `samples`: 16 kHz mono f32. Returns the decoded transcript.
	pub fn transcribe(&mut self, samples: &[f32]) -> Result<String, String> {
		if samples.is_empty() {
			return Ok(String::new());
		}
		let segments = self.vad.speech_segments(samples, VAD_MIN_DURATION_SECS, MAX_SEGMENT_SECS);
		let mut parts: Vec<String> = Vec::with_capacity(segments.len());
		for range in &segments {
			let text = self.transcribe_segment(&samples[range.clone()])?;
			if !text.is_empty() {
				parts.push(text);
			}
		}
		Ok(parts.join(" "))
	}

	/// One piece through preprocessor -> model -> CTC greedy decode.
	fn transcribe_segment(&mut self, samples: &[f32]) -> Result<String, String> {
		let len = samples.len();
		let waveforms =
			Tensor::from_array((vec![1usize, len], samples.to_vec())).map_err(|e| e.to_string())?;
		let waveforms_lens =
			Tensor::from_array((vec![1usize], vec![len as i64])).map_err(|e| e.to_string())?;
		let mel_out = self
			.mel
			.run(ort::inputs![
				"waveforms" => waveforms,
				"waveforms_lens" => waveforms_lens
			])
			.map_err(|e| format!("preprocessor failed: {e}"))?;
		let (_, features) =
			mel_out["features"].try_extract_tensor::<f32>().map_err(|e| e.to_string())?;
		let (_, features_lens) =
			mel_out["features_lens"].try_extract_tensor::<i64>().map_err(|e| e.to_string())?;
		// Batch is always 1; the graph's own lens is authoritative.
		let t = features_lens[0] as usize;
		let features = Tensor::from_array((vec![1usize, 64, t], features[..64 * t].to_vec()))
			.map_err(|e| e.to_string())?;
		let feature_lengths =
			Tensor::from_array((vec![1usize], vec![t as i64])).map_err(|e| e.to_string())?;
		let out = self
			.model
			.run(ort::inputs![
				"features" => features,
				"feature_lengths" => feature_lengths
			])
			.map_err(|e| format!("inference failed: {e}"))?;
		let (log_probs_shape, log_probs) =
			out["log_probs"].try_extract_tensor::<f32>().map_err(|e| e.to_string())?;
		// Shape [1, T', 257]; argmax ignores the extra frames past the
		// length anyway, so decode the model's full output.
		let frames = log_probs_shape[1] as usize;
		let classes = log_probs_shape[2] as usize;
		Ok(decode_ctc(log_probs, frames, classes, &self.vocab))
	}
}

/// Parses the vocab file: one `token id` pair per line, 257 entries, ids
/// matching the model's output classes (blank = 256). `▁` becomes a space.
fn load_vocab(path: &Path) -> Result<Vec<String>, String> {
	let text =
		std::fs::read_to_string(path).map_err(|e| format!("cannot read vocab {}: {e}", path.display()))?;
	let mut vocab: Vec<String> = Vec::new();
	for line in text.lines() {
		let Some((token, id)) = line.trim_end().rsplit_once(' ') else {
			continue;
		};
		let id: usize = id.parse().map_err(|e| format!("bad vocab line {line:?}: {e}"))?;
		if id == BLANK_ID && token != "<blk>" {
			return Err(format!("vocab entry {id} is {token:?}, expected <blk>"));
		}
		if vocab.len() < id + 1 {
			vocab.resize(id + 1, String::new());
		}
		vocab[id] = token.replace(SPACE_TOKEN, " ");
	}
	if vocab.len() != BLANK_ID + 1 {
		return Err(format!("vocab has {} entries, expected {}", vocab.len(), BLANK_ID + 1));
	}
	Ok(vocab)
}

/// CTC greedy decode: argmax per frame, drop blanks, collapse consecutive
/// repeats, map ids to tokens.
fn decode_ctc(log_probs: &[f32], frames: usize, classes: usize, vocab: &[String]) -> String {
	let mut text = String::new();
	let mut last: Option<usize> = None;
	for frame in 0..frames {
		let row = &log_probs[frame * classes..(frame + 1) * classes];
		let arg = row
			.iter()
			.enumerate()
			.max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
			.map(|(i, _)| i)
			.unwrap_or(BLANK_ID);
		if arg != BLANK_ID && last != Some(arg) {
			if let Some(token) = vocab.get(arg) {
				text.push_str(token);
			}
		}
		last = Some(arg);
	}
	text.trim().to_string()
}

#[cfg(test)]
mod tests {
	use super::*;

	fn vocab() -> Vec<String> {
		let mut v = vec![String::new(); 257];
		v[0] = "<unk>".into();
		v[1] = " ".into();
		v[2] = "а".into();
		v[3] = "б".into();
		v[256] = "<blk>".into();
		v
	}

	#[test]
	fn decode_drops_blanks_and_collapses_repeats() {
		// Frames: blank, а, а (repeat), blank, б, а -> "аба".
		let mut probs = Vec::new();
		for arg in [256usize, 2, 2, 256, 3, 2] {
			for c in 0..257 {
				probs.push(if c == arg { 0.0 } else { -10.0 });
			}
		}
		assert_eq!(decode_ctc(&probs, 6, 257, &vocab()), "аба");
	}

	#[test]
	fn decode_maps_space_tokens_and_trims() {
		// ▁а▁б -> " а б" -> trimmed "а б". Consecutive ▁ cannot occur
		// (repeats collapse at the argmax level), so no space squeezing is
		// needed anywhere.
		let mut probs = Vec::new();
		for arg in [1usize, 2, 1, 3] {
			for c in 0..257 {
				probs.push(if c == arg { 0.0 } else { -10.0 });
			}
		}
		assert_eq!(decode_ctc(&probs, 4, 257, &vocab()), "а б");
	}

	#[test]
	fn decode_returns_empty_for_all_blank() {
		let probs = vec![-1.0f32; 3 * 257];
		assert_eq!(decode_ctc(&probs, 3, 257, &vocab()), "");
	}
}
