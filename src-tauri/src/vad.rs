//! Silero VAD (vendored, MIT): splits long recordings into speech islands
//! for engines that should not take a long recording in one piece (see
//! DESIGN.md, "Engines").

use std::ops::Range;

use ort::session::Session;
use ort::value::Tensor;

/// Silero VAD v5.1.2 (MIT): `input [1, 512]` f32 + `state [2, 1, 128]` +
/// `sr` i64 -> `output [1, 1]` speech probability + `stateN`.
const VAD_GRAPH: &[u8] = include_bytes!("../assets/silero_vad.onnx");

/// VAD chunk size in samples at 16 kHz (32 ms) — the granularity of the
/// speech/silence decision and of segment boundaries.
const VAD_CHUNK: usize = 512;

/// Speech probability above which a chunk counts as speech (silero default).
const VAD_THRESHOLD: f32 = 0.5;

/// Speech islands shorter than this are dropped (clicks, breaths).
const VAD_MIN_SPEECH_SECS: f32 = 0.25;

/// Silence gaps shorter than this do not split a segment.
const VAD_MIN_SILENCE_SECS: f32 = 0.3;

/// Padding added around each speech island, so words are not clipped at
/// the boundary.
const VAD_PAD_SECS: f32 = 0.15;

pub struct Vad {
	session: Session,
}

impl Vad {
	pub fn new() -> Result<Self, String> {
		let session = Session::builder()
			.map_err(|e| format!("cannot create session builder: {e}"))?
			.commit_from_memory(VAD_GRAPH)
			.map_err(|e| format!("cannot load VAD model: {e}"))?;
		Ok(Self { session })
	}

	/// Splits a recording into sample ranges to transcribe. Recordings up to
	/// `whole_below_secs` (the dictation norm) skip VAD entirely; longer ones
	/// are cut into speech islands, each capped at `max_segment_secs`.
	pub fn speech_segments(
		&mut self,
		samples: &[f32],
		whole_below_secs: f32,
		max_segment_secs: f32,
	) -> Vec<Range<usize>> {
		let vad_limit = (whole_below_secs * crate::audio::SAMPLE_RATE as f32) as usize;
		let max = (max_segment_secs * crate::audio::SAMPLE_RATE as f32) as usize;
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
			.flat_map(|range| pad_and_split(range, samples.len(), max))
			.collect();
		if ranges.is_empty() {
			// VAD found nothing above the threshold but the recording passed
			// the pipeline's silence guard: fall back to fixed pieces rather
			// than dropping the dictation (VAD pathology, not user silence).
			ranges = fixed_pieces(samples.len(), max);
			eprintln!("vad: found no speech in {} samples; transcribing whole audio", samples.len());
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
			let Ok(out) = self.session.run(ort::inputs![
				"input" => input,
				"state" => state_tensor,
				"sr" => sr
			]) else {
				// One failed chunk should not kill the dictation; treat as
				// silence and keep the previous state.
				eprintln!("vad: chunk failed; treating as silence");
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
/// than `max` samples.
fn pad_and_split(range: Range<usize>, total: usize, max: usize) -> Vec<Range<usize>> {
	let pad = (VAD_PAD_SECS * crate::audio::SAMPLE_RATE as f32) as usize;
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
fn fixed_pieces(total: usize, max: usize) -> Vec<Range<usize>> {
	(0..total).step_by(max).map(|s| s..(s + max).min(total)).collect()
}

#[cfg(test)]
mod tests {
	use super::*;

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
		let pieces = pad_and_split(0..480_000, 500_000, 320_000);
		assert!(pieces.iter().all(|p| p.end - p.start <= 320_000));
		assert_eq!(pieces.first().unwrap().start, 0);
		assert_eq!(pieces.last().unwrap().end, 480_000 + 2400);
		assert_eq!(pieces.len(), 2);
	}

	#[test]
	fn fixed_pieces_cover_everything() {
		let pieces = fixed_pieces(700_000, 320_000);
		assert_eq!(pieces.first().unwrap().start, 0);
		assert_eq!(pieces.last().unwrap().end, 700_000);
		assert!(pieces.iter().all(|p| p.end - p.start <= 320_000));
	}
}
