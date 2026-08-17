//! Microphone capture via cpal (default input device, PipeWire-ALSA path).
//!
//! Design (DESIGN.md, "Pipeline"): record at the device rate, convert to
//! 16 kHz mono f32 for whisper. `Recorder` owns the cpal input stream and an
//! accumulating sample buffer; `stop()` drops the stream and returns the
//! samples resampled to 16 kHz mono.

use std::sync::{Arc, Mutex};

/// Averages interleaved multi-channel samples into a mono signal.
/// A trailing partial frame is dropped.
pub fn to_mono(samples: &[f32], channels: usize) -> Vec<f32> {
	match channels {
		0 | 1 => samples.to_vec(),
		_ => samples
			.chunks_exact(channels)
			.map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
			.collect(),
	}
}

/// Whisper input rate (DESIGN.md, "Pipeline").
pub const WHISPER_SAMPLE_RATE: u32 = 16000;

/// Resamples a mono signal between sample rates.
///
/// Integer ratios (e.g. PipeWire's 48 kHz -> 16 kHz) decimate with a box
/// filter; other ratios use linear interpolation. Speech recognition
/// tolerates both; this avoids pulling in a resampler dependency.
pub fn resample(input: &[f32], from_hz: u32, to_hz: u32) -> Vec<f32> {
	if input.is_empty() || from_hz == to_hz {
		return input.to_vec();
	}
	let ratio = from_hz as f64 / to_hz as f64;
	let out_len = (input.len() as f64 / ratio).floor() as usize;
	let mut out = Vec::with_capacity(out_len);
	if from_hz % to_hz == 0 {
		let window = (from_hz / to_hz) as usize;
		for chunk in input.chunks_exact(window) {
			out.push(chunk.iter().sum::<f32>() / window as f32);
		}
	} else {
		for i in 0..out_len {
			let pos = i as f64 * ratio;
			let left = pos.floor() as usize;
			let frac = (pos - left as f64) as f32;
			let a = input[left];
			let b = input.get(left + 1).copied().unwrap_or(a);
			out.push(a + (b - a) * frac);
		}
	}
	out
}

/// Opens the default input device and returns its name, or an error.
pub fn default_input_name() -> Result<String, String> {
	use cpal::traits::{DeviceTrait, HostTrait};

	let host = cpal::default_host();
	let device = host
		.default_input_device()
		.ok_or("no default input device")?;
	Ok(device.name().map_err(|e| e.to_string())?)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn to_mono_mixes_stereo_to_single_channel() {
		// Interleaved stereo frames (1.0, -1.0) and (0.5, 1.5) average to
		// 0.0 and 1.0.
		let stereo = [1.0, -1.0, 0.5, 1.5];
		assert_eq!(to_mono(&stereo, 2), vec![0.0, 1.0]);
	}

	#[test]
	fn to_mono_passes_mono_through_unchanged() {
		let mono = [0.25, -0.5, 0.75];
		assert_eq!(to_mono(&mono, 1), mono.to_vec());
	}

	#[test]
	fn resample_returns_input_unchanged_when_rates_match() {
		let input = [0.1, -0.2, 0.3];
		assert_eq!(resample(&input, 16000, 16000), input.to_vec());
	}

	#[test]
	fn resample_decimates_integer_ratio_by_averaging() {
		// 48 kHz -> 16 kHz: every 3-sample window becomes one sample.
		let input = [1.0, 2.0, 3.0, 3.0, 3.0, 3.0];
		assert_eq!(resample(&input, 48000, 16000), vec![2.0, 3.0]);
		// A constant signal stays constant and shrinks by the ratio.
		let constant = [0.5; 90];
		assert_eq!(resample(&constant, 48000, 16000), vec![0.5; 30]);
	}

	#[test]
	fn resample_interpolates_non_integer_ratio() {
		// 44.1 kHz -> 16 kHz: linear interpolation of a constant signal must
		// stay constant; output length is floor(n * 16000 / 44100).
		let n = 4410;
		let constant = vec![0.25; n];
		let out = resample(&constant, 44100, 16000);
		assert_eq!(out.len(), 1600);
		assert!(out.iter().all(|&s| (s - 0.25).abs() < 1e-6));
	}

	#[test]
	fn resample_returns_empty_for_empty_input() {
		assert!(resample(&[], 48000, 16000).is_empty());
		assert!(resample(&[], 44100, 16000).is_empty());
	}

	#[test]
	fn default_input_device_resolves_with_a_name() {
		// Live check of the cpal/PipeWire path (DESIGN.md risk #3).
		let name = default_input_name().expect("default input device");
		assert!(!name.is_empty());
	}
}

/// Opens an input stream whose callback converts interleaved frames to
/// mono f32 and appends them to the shared buffer.
fn open_stream<T>(
	device: &cpal::Device,
	config: &cpal::StreamConfig,
	samples: Arc<Mutex<Vec<f32>>>,
	channels: usize,
) -> Result<cpal::Stream, String>
where
	T: cpal::SizedSample,
	f32: cpal::FromSample<T>,
{
	use cpal::traits::DeviceTrait;
	use cpal::Sample;

	let data = move |data: &[T], _: &cpal::InputCallbackInfo| {
		let converted: Vec<f32> = data.iter().map(|&s| f32::from_sample(s)).collect();
		let mut buffer = samples.lock().unwrap();
		buffer.extend(to_mono(&converted, channels));
	};
	device
		.build_input_stream::<T, _, _>(config, data, |e| eprintln!("audio input error: {e}"), None)
		.map_err(|e| e.to_string())
}

pub struct Recorder {
	stream: cpal::Stream,
	samples: Arc<Mutex<Vec<f32>>>,
	sample_rate: u32,
}

impl Recorder {
	/// Starts capturing from the default input device at its native rate.
	pub fn start() -> Result<Self, String> {
		use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

		let host = cpal::default_host();
		let device = host
			.default_input_device()
			.ok_or("no default input device")?;
		let supported = device.default_input_config().map_err(|e| e.to_string())?;
		let channels = supported.channels();
		let sample_rate = supported.sample_rate();
		let config = cpal::StreamConfig {
			channels,
			sample_rate,
			buffer_size: cpal::BufferSize::Default,
		};
		let samples = Arc::new(Mutex::new(Vec::new()));
		// Dispatch over the runtime sample format to the generic stream
		// builder; one arm per `SampleFormat` variant.
		macro_rules! open {
			($sample:ty) => {
				open_stream::<$sample>(&device, &config, Arc::clone(&samples), channels as usize)
			};
		}
		let stream = match supported.sample_format() {
			cpal::SampleFormat::F32 => open!(f32),
			cpal::SampleFormat::F64 => open!(f64),
			cpal::SampleFormat::I8 => open!(i8),
			cpal::SampleFormat::I16 => open!(i16),
			cpal::SampleFormat::I32 => open!(i32),
			cpal::SampleFormat::I64 => open!(i64),
			cpal::SampleFormat::U8 => open!(u8),
			cpal::SampleFormat::U16 => open!(u16),
			cpal::SampleFormat::U32 => open!(u32),
			cpal::SampleFormat::U64 => open!(u64),
			cpal::SampleFormat::I24 => open!(cpal::I24),
			_ => return Err("unsupported input sample format".into()),
		}?;
		stream.play().map_err(|e| e.to_string())?;
		Ok(Self {
			stream,
			samples,
			sample_rate: sample_rate.0,
		})
	}

	/// Stops recording and returns captured samples at 16 kHz mono f32.
	pub fn stop(self) -> Result<Vec<f32>, String> {
		drop(self.stream);
		let mut buffer = self.samples.lock().unwrap();
		let samples = std::mem::take(&mut *buffer);
		Ok(resample(&samples, self.sample_rate, WHISPER_SAMPLE_RATE))
	}
}
