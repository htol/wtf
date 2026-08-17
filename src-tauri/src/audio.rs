//! Microphone capture via cpal (default input device, PipeWire-ALSA path).
//!
//! Design (DESIGN.md, "Pipeline"): record at the device rate, convert to
//! 16 kHz mono f32 for whisper. `Recorder` will own the cpal input stream
//! and an accumulating sample buffer; `stop()` returns the samples.

/// Opens the default input device and returns its name, or an error.
/// Skeleton: verifies the cpal/PipeWire path works on this machine.
pub fn default_input_name() -> Result<String, String> {
	use cpal::traits::{DeviceTrait, HostTrait};

	let host = cpal::default_host();
	let device = host
		.default_input_device()
		.ok_or("no default input device")?;
	Ok(device.name().map_err(|e| e.to_string())?)
}

pub struct Recorder;

impl Recorder {
	pub fn start() -> Result<Self, String> {
		// TODO: open exclusive input stream, push f32 samples into a buffer
		// shared with `stop()`. Resample to 16 kHz mono.
		Err("Recorder::start not implemented".into())
	}

	/// Stops recording and returns captured samples at 16 kHz mono f32.
	pub fn stop(self) -> Result<Vec<f32>, String> {
		// TODO: drop stream, drain buffer.
		Err("Recorder::stop not implemented".into())
	}
}
