//! GigaAM v3 e2e-CTC: the second ASR engine next to whisper (see DESIGN.md,
//! "Engines"). Russian speech, CPU only, via ONNX Runtime.
//!
//! Three `ort` sessions per engine instance:
//! - log-mel preprocessor graph, vendored from onnx-asr (MIT) and run as-is —
//!   zero parity risk with the reference Python toolchain;
//! - the acoustic model (`v3_e2e_ctc[.int8].onnx`, downloaded to the models
//!   dir by `models::download_gigaam_model`);
//! - Silero VAD (MIT), used to split long recordings: the model's segment
//!   limit is ~25 s, so anything longer is cut into speech islands first.
//!
//! Decoding is CTC greedy: argmax per 40 ms frame, drop blanks, collapse
//! repeats, map ids to tokens. Verified byte-identical to `onnx_asr.recognize`
//! on the official GigaAM sample.

use std::ops::Range;
use std::path::Path;

use ort::session::Session;
use ort::value::Tensor;

/// Torchaudio-style log-mel graph: `waveforms [B, L]` f32 @16 kHz +
/// `waveforms_lens [B]` i64 -> `features [B, 64, T]` + `features_lens [B]`.
/// Extracted from the onnx-asr wheel (MIT), revision pinned by the file
/// itself; see DESIGN.md "Engines".
const MEL_GRAPH: &[u8] = include_bytes!("../assets/gigaam_mel.onnx");

/// Silero VAD v5.1.2 (MIT): `input [1, 512]` f32 + `state [2, 1, 128]` +
/// `sr` i64 -> `output [1, 1]` speech probability + `stateN`.
const VAD_GRAPH: &[u8] = include_bytes!("../assets/silero_vad.onnx");

/// VAD chunk size in samples at 16 kHz (32 ms) — the granularity of the
/// speech/silence decision and of segment boundaries.
const VAD_CHUNK: usize = 512;

/// Speech probability above which a chunk counts as speech (silero default).
const VAD_THRESHOLD: f32 = 0.5;

/// Recordings shorter than this are transcribed whole, without VAD.
const VAD_MIN_DURATION_SECS: f32 = 20.0;

/// Speech islands shorter than this are dropped (clicks, breaths).
const VAD_MIN_SPEECH_SECS: f32 = 0.25;

/// Silence gaps shorter than this do not split a segment.
const VAD_MIN_SILENCE_SECS: f32 = 0.3;

/// Padding added around each speech island, so words are not clipped at
/// the boundary.
const VAD_PAD_SECS: f32 = 0.15;

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
	vad: Session,
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
		let vad = builder()?
			.commit_from_memory(VAD_GRAPH)
			.map_err(|e| format!("cannot load VAD model: {e}"))?;
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
		let segments = self.speech_segments(samples);
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

	/// Splits a recording into sample ranges to transcribe. Short recordings
	/// (the dictation norm) skip VAD entirely; long ones are cut into speech
	/// islands, each capped at `MAX_SEGMENT_SECS`.
	fn speech_segments(&mut self, samples: &[f32]) -> Vec<Range<usize>> {
		let vad_limit = (VAD_MIN_DURATION_SECS * crate::audio::SAMPLE_RATE as f32) as usize;
		if samples.len() <= vad_limit {
			return vec![0..samples.len()];
		}
		let probs = self.speech_probabilities(samples);
		let islands = speech_islands(&prob_flags(&probs));
		let mut ranges: Vec<Range<usize>> = islands
			.iter()
			.map(|island| {
				let start = island.start.saturating_mul(VAD_CHUNK);
				let end = (island.end * VAD_CHUNK).min(samples.len());
				start..end
			})
			.flat_map(|range| pad_and_split(range, samples.len()))
			.collect();
		if ranges.is_empty() {
			// VAD found nothing above the threshold but the recording passed
			// the pipeline's silence guard: fall back to fixed pieces rather
			// than dropping the dictation (VAD pathology, not user silence).
			ranges = fixed_pieces(samples.len());
			eprintln!("gigaam: VAD found no speech in {} samples; transcribing whole audio", samples.len());
		}
		debug_assert!(ranges.iter().all(|r| r.start < r.end && r.end <= samples.len()));
		ranges
	}

	/// Runs Silero VAD over the whole recording, one 512-sample chunk at a
	/// time, carrying the streaming state (the official usage pattern).
	fn speech_probabilities(&mut self, samples: &[f32]) -> Vec<f32> {
		let mut state = vec![0.0f32; 2 * 128];
		let mut probs = Vec::with_capacity(samples.len().div_ceil(VAD_CHUNK));
		let mut chunk = vec![0.0f32; VAD_CHUNK];
		for piece in samples.chunks(VAD_CHUNK) {
			chunk.fill(0.0);
			chunk[..piece.len()].copy_from_slice(piece);
			// `Value` is not `Clone`: build the small inputs fresh each chunk.
			let input =
				Tensor::from_array((vec![1usize, VAD_CHUNK], chunk.clone())).expect("vad input");
			let state_tensor =
				Tensor::from_array((vec![2usize, 1, 128], state.clone())).expect("vad state");
			let sr = Tensor::from_array((Vec::<usize>::new(), vec![crate::audio::SAMPLE_RATE as i64]))
				.expect("scalar rate tensor");
			let Ok(out) = self.vad.run(ort::inputs![
				"input" => input,
				"state" => state_tensor,
				"sr" => sr
			]) else {
				// One failed chunk should not kill the dictation; treat as
				// silence and keep the previous state.
				eprintln!("gigaam: VAD chunk failed; treating as silence");
				probs.push(0.0);
				continue;
			};
			if let (Ok((_, p)), Ok((_, s))) = (
				out["output"].try_extract_tensor::<f32>(),
				out["stateN"].try_extract_tensor::<f32>(),
			) {
				probs.push(p[0]);
				state.copy_from_slice(s);
			} else {
				probs.push(0.0);
			}
		}
		probs
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

/// Chunk-index ranges of continuous speech after dropping too-short islands
/// and bridging short silence gaps.
fn speech_islands(flags: &[bool]) -> Vec<Range<usize>> {
	let min_speech = (VAD_MIN_SPEECH_SECS * crate::audio::SAMPLE_RATE as f32) as usize / VAD_CHUNK;
	let min_silence = (VAD_MIN_SILENCE_SECS * crate::audio::SAMPLE_RATE as f32) as usize / VAD_CHUNK;
	let mut kept: Vec<bool> = flags.to_vec();
	// Drop speech islands shorter than the minimum.
	let mut island = 0;
	while island < kept.len() {
		if !kept[island] {
			island += 1;
			continue;
		}
		let end = kept[island..].iter().position(|&s| !s).map(|n| island + n).unwrap_or(kept.len());
		if end - island < min_speech.max(1) {
			for f in kept[island..end].iter_mut() {
				*f = false;
			}
		}
		island = end;
	}
	// Merge islands separated by less than the minimum silence.
	let mut ranges: Vec<Range<usize>> = Vec::new();
	let mut iter = kept.iter().enumerate().filter(|&(_, &s)| s).peekable();
	while let Some((start, _)) = iter.next() {
		let mut end = start + 1;
		while let Some(&(next, _)) = iter.peek() {
			if next - end < min_silence.max(1) {
				end = next + 1;
				iter.next();
			} else {
				break;
			}
		}
		ranges.push(start..end);
	}
	ranges
}

fn prob_flags(probs: &[f32]) -> Vec<bool> {
	probs.iter().map(|&p| p >= VAD_THRESHOLD).collect()
}

/// Pads a segment around its boundaries and splits it into pieces no longer
/// than `MAX_SEGMENT_SECS`.
fn pad_and_split(range: Range<usize>, total: usize) -> Vec<Range<usize>> {
	let pad = (VAD_PAD_SECS * crate::audio::SAMPLE_RATE as f32) as usize;
	let max = (MAX_SEGMENT_SECS * crate::audio::SAMPLE_RATE as f32) as usize;
	let start = range.start.saturating_sub(pad);
	let end = (range.end + pad).min(total);
	let mut pieces = Vec::new();
	let mut s = start;
	while s < end {
		let e = (s + max).min(end);
		pieces.push(s..e);
		s = e;
	}
	pieces
}

/// Whole-audio fallback for VAD-pathology: fixed max-size pieces.
fn fixed_pieces(total: usize) -> Vec<Range<usize>> {
	let max = (MAX_SEGMENT_SECS * crate::audio::SAMPLE_RATE as f32) as usize;
	(0..total).step_by(max).map(|s| s..(s + max).min(total)).collect()
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

	#[test]
	fn islands_drop_short_and_bridge_small_gaps() {
		// 20 chunks speech (>= min_speech 7), 5 gap (< min_silence 9),
		// 20 speech -> merged into one island; then a 2-chunk blip far
		// away is dropped.
		let mut flags = vec![false; 64];
		for f in flags.iter_mut().take(20) {
			*f = true;
		}
		for f in flags.iter_mut().skip(25).take(20) {
			*f = true;
		}
		flags[50] = true;
		flags[51] = true;
		assert_eq!(speech_islands(&flags), vec![0..45]);
	}

	#[test]
	fn islands_keep_long_separate_segments() {
		// 20 speech, 15 silence (>= 9), 20 speech: two islands.
		let mut flags = vec![false; 64];
		for f in flags.iter_mut().take(20) {
			*f = true;
		}
		for f in flags.iter_mut().skip(35).take(20) {
			*f = true;
		}
		assert_eq!(speech_islands(&flags), vec![0..20, 35..55]);
	}

	#[test]
	fn pad_and_split_caps_piece_length() {
		// 30 s of speech: padded by ±150 ms, then split into 20 s + rest.
		let pieces = pad_and_split(0..480_000, 500_000);
		assert!(pieces.iter().all(|p| p.end - p.start <= 320_000));
		assert_eq!(pieces.first().unwrap().start, 0);
		assert_eq!(pieces.last().unwrap().end, 480_000 + 2400);
		assert_eq!(pieces.len(), 2);
	}

	#[test]
	fn fixed_pieces_cover_everything() {
		let pieces = fixed_pieces(700_000);
		assert_eq!(pieces.first().unwrap().start, 0);
		assert_eq!(pieces.last().unwrap().end, 700_000);
		assert!(pieces.iter().all(|p| p.end - p.start <= 320_000));
	}
}
