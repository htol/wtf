//! Dictation pipeline: `record` hotkey toggle -> capture -> transcribe ->
//! paste into the focused app -> history (DESIGN.md "Pipeline").
//!
//! Engine routing (DESIGN.md, "Engines") follows the `engine` setting.
//! GigaAM is Russian only and ignores the language; without its model it
//! falls back to whisper with a one-time download suggestion. Qwen3-ASR
//! without a model fails like whisper does (settings window opens).
//!
//! `Dictation` is app-managed state. Engines are created lazily on the first
//! transcription and cached, each slot keyed by model path (loading is
//! expensive); all stay loaded across engine switches.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use tauri::{Emitter, Manager};

use crate::{asr, audio, history, inject, models, settings};

pub struct Dictation {
	recorder: Mutex<Option<audio::Recorder>>,
	/// Whisper slot: Transcriber::Whisper, keyed by ggml model path.
	whisper: Mutex<Option<(PathBuf, asr::Transcriber)>>,
	/// GigaAM slot: Transcriber::GigaAm, keyed by onnx model path.
	gigaam: Mutex<Option<(PathBuf, asr::Transcriber)>>,
	/// Qwen3-ASR slot: Transcriber::Qwen, keyed by model dir.
	qwen: Mutex<Option<(PathBuf, asr::Transcriber)>>,
	/// Guard for the one-time-per-session GigaAM download suggestion.
	gigaam_suggested: AtomicBool,
}

impl Dictation {
	pub fn new() -> Self {
		Self {
			recorder: Mutex::new(None),
			whisper: Mutex::new(None),
			gigaam: Mutex::new(None),
			qwen: Mutex::new(None),
			gigaam_suggested: AtomicBool::new(false),
		}
	}
}

/// Handles a `record` activation: starts or stops recording. Stopping runs
/// the rest of the pipeline (transcribe -> paste -> history) on a blocking
/// task so the portal activation stream is never blocked.
pub fn toggle_record(app: &tauri::AppHandle) {
	let state = app.state::<Dictation>();
	let mut slot = state.recorder.lock().unwrap();
	if let Some(recorder) = slot.take() {
		drop(slot); // don't hold the lock while stopping the stream
		crate::tray::set_recording(app, false);
		let _ = app.emit("recording", false);
		let app = app.clone();
		tauri::async_runtime::spawn_blocking(move || {
			// Model load + inference + resample take seconds: all blocking.
			let outcome = recorder
				.stop()
				.map_err(|e| format!("record stop failed: {e}"))
				.and_then(|samples| {
				// Silence guard: whisper hallucinates fluent text on near-silence.
				// Threshold comes from settings; 0 disables the check.
				let threshold = settings::load().silence_peak;
				let peak = audio::peak(&samples);
				if threshold > 0.0 && peak < threshold {
					eprintln!("no speech detected (peak {peak:.3} < {threshold:.3})");
					let _ = app.emit("no-speech", ());
					return Ok(());
				}
				let _ = app.emit("processing", true);
					let result = transcribe_and_paste(&app, &samples);
					let _ = app.emit("processing", false);
					result
				});
			hide_overlay(&app, outcome.is_err());
			if let Err(e) = outcome {
				eprintln!("pipeline failed: {e}");
				let _ = app.emit("pipeline-error", e);
			}
		});
		return;
	}
	match audio::Recorder::start(settings::load().input_device) {
		Ok(recorder) => {
			*slot = Some(recorder);
			crate::tray::set_recording(app, true);
			show_overlay(app);
			spawn_level_ticker(app.clone());
			let _ = app.emit("recording", true);
		}
		Err(e) => {
			eprintln!("record start failed: {e}");
			let _ = app.emit("pipeline-error", format!("record start failed: {e}"));
		}
	}
}

/// Prepares the overlay window without mapping it. GTK3 on Wayland cannot
/// resize a mapped toplevel (`gtk_window_resize` is silently ignored after
/// map — the compositor owns the size), so the overlay is shown/hidden
/// instead of collapsed to 1x1. `set_focusable(false)` must run before the
/// first `show()`. On Wayland the window becomes a layer-shell surface
/// (see `overlay_layer`): a plain xdg toplevel has no way to refuse focus,
/// stay on top or pick its position, and every map would activate it.
pub fn prime_overlay(app: &tauri::AppHandle) {
	let Some(window) = app.get_webview_window("overlay") else {
		return;
	};
	let _ = window.set_focusable(false);
	let _ = window.set_always_on_top(true);
	#[cfg(target_os = "linux")]
	overlay_layer::init(&window);
	// macOS has no compositor rule to place the overlay: restore the saved
	// position (fractions of the monitor, see Overlay.svelte) and follow the
	// user across Spaces.
	#[cfg(target_os = "macos")]
	{
		let _ = window.set_visible_on_all_workspaces(true);
		if let Ok(Some(monitor)) = window.primary_monitor() {
			let settings = settings::load();
			let size = monitor.size();
			let _ = window.set_position(tauri::PhysicalPosition::new(
				settings.overlay_x * size.width as f64,
				settings.overlay_y * size.height as f64,
			));
		}
	}
}

/// Linux/Wayland: the overlay as a wlr-layer-shell surface. The compositor
/// never gives it keyboard focus (`KeyboardMode::None`), keeps it in the
/// overlay layer and places it by the margins the app sets — so no KWin or
/// Hyprland window rules are needed and the map on each recording cannot
/// steal focus from the dictation target. The position is kept as fractions
/// of the monitor (`overlay_x`/`overlay_y`, shared with macOS); dragging the
/// pill moves it through `shift_overlay`, since a layer surface cannot be
/// moved by the compositor. On X11 the library reports no support and the
/// window stays a plain toplevel.
#[cfg(target_os = "linux")]
mod overlay_layer {
	use std::sync::atomic::{AtomicBool, Ordering};

	use gtk::gdk;
	use gtk::prelude::*;
	use gtk_layer_shell::{Edge, KeyboardMode, Layer, LayerShell};

	use crate::settings;

	static ACTIVE: AtomicBool = AtomicBool::new(false);

	pub fn active() -> bool {
		ACTIVE.load(Ordering::Acquire)
	}

	/// Must run on the main thread before the window is first mapped:
	/// gtk-layer-shell can only take over an unrealized GtkWindow.
	pub fn init(window: &tauri::WebviewWindow) {
		if !gtk_layer_shell::is_supported() {
			return;
		}
		let Ok(gtk) = window.gtk_window() else {
			return;
		};
		gtk.init_layer_shell();
		gtk.set_namespace("wtf-overlay");
		gtk.set_layer(Layer::Overlay);
		gtk.set_keyboard_mode(KeyboardMode::None);
		gtk.set_anchor(Edge::Left, true);
		gtk.set_anchor(Edge::Top, true);
		// Pin the surface to one output so the margins and the saved
		// fractions refer to the same monitor.
		if let Some(monitor) = home_monitor(&gtk) {
			gtk.set_monitor(&monitor);
			let settings = settings::load();
			let geometry = monitor.geometry();
			gtk.set_layer_shell_margin(
				Edge::Left,
				(settings.overlay_x * geometry.width() as f64) as i32,
			);
			gtk.set_layer_shell_margin(
				Edge::Top,
				(settings.overlay_y * geometry.height() as f64) as i32,
			);
		}
		// Warm-up map: into the first layer surface of this window GTK never
		// draws a frame (the pill maps but stays empty, and WebKit later
		// paints it at the wrong size); a surface mapped after a configured
		// and unmapped one draws right after configure. Realizing ahead or
		// an unmap before the configure arrives do not help, so the window
		// stays mapped briefly. Nothing is drawn into it, so nothing
		// flashes.
		let _ = window.show();
		let warm = window.clone();
		std::thread::spawn(move || {
			std::thread::sleep(std::time::Duration::from_millis(300));
			let _ = warm.hide();
		});
		ACTIVE.store(true, Ordering::Release);
	}

	/// The monitor the overlay lives on. Wayland has no primary-monitor
	/// notion GDK could report, and GDK's monitor 0 is merely the output
	/// the compositor advertised first; the layout origin is where KDE and
	/// most compositors put the primary screen, so prefer the monitor that
	/// contains it.
	fn home_monitor(gtk: &gtk::ApplicationWindow) -> Option<gdk::Monitor> {
		let display = gtk.display();
		if let Some(primary) = display.primary_monitor() {
			return Some(primary);
		}
		(0..display.n_monitors())
			.filter_map(|i| display.monitor(i))
			.find(|m| {
				let g = m.geometry();
				g.x() <= 0 && 0 < g.x() + g.width() && g.y() <= 0 && 0 < g.y() + g.height()
			})
			.or_else(|| display.monitor(0))
	}

	/// Moves the surface by `dx`/`dy` logical pixels, clamped to the
	/// monitor, and optionally persists the new position as fractions.
	pub fn shift(window: &tauri::WebviewWindow, dx: i32, dy: i32, persist: bool) {
		let Ok(gtk) = window.gtk_window() else {
			return;
		};
		let Some(monitor) = <gtk::ApplicationWindow as LayerShell>::monitor(&gtk).or_else(|| home_monitor(&gtk)) else {
			return;
		};
		let geometry = monitor.geometry();
		let (width, height) = gtk.size();
		let max_x = (geometry.width() - width).max(0);
		let max_y = (geometry.height() - height).max(0);
		let x = (gtk.layer_shell_margin(Edge::Left) + dx).clamp(0, max_x);
		let y = (gtk.layer_shell_margin(Edge::Top) + dy).clamp(0, max_y);
		gtk.set_layer_shell_margin(Edge::Left, x);
		gtk.set_layer_shell_margin(Edge::Top, y);
		if persist {
			let _ = settings::set_overlay_position(
				x as f64 / geometry.width().max(1) as f64,
				y as f64 / geometry.height().max(1) as f64,
			);
		}
	}
}

/// Whether the overlay is a layer-shell surface (Linux/Wayland), in which
/// case the overlay window moves itself through `shift_overlay` instead of
/// the compositor's drag.
#[tauri::command]
pub fn overlay_is_layer() -> bool {
	#[cfg(target_os = "linux")]
	{
		overlay_layer::active()
	}
	#[cfg(not(target_os = "linux"))]
	{
		false
	}
}

/// Drag step from the overlay window (logical pixels); `persist` on the
/// final step saves the position. No-op unless the overlay is a layer
/// surface. GTK calls must run on the main thread; the command returns
/// only once the step is applied, so the overlay can pace its steps on
/// the surface actually having moved.
#[tauri::command]
pub fn shift_overlay(app: tauri::AppHandle, dx: i32, dy: i32, persist: bool) {
	#[cfg(target_os = "linux")]
	{
		if !overlay_layer::active() {
			return;
		}
		let (done, applied) = std::sync::mpsc::channel();
		let handle = app.clone();
		let _ = app.run_on_main_thread(move || {
			if let Some(window) = handle.get_webview_window("overlay") {
				overlay_layer::shift(&window, dx, dy, persist);
			}
			let _ = done.send(());
		});
		let _ = applied.recv_timeout(std::time::Duration::from_millis(100));
	}
	#[cfg(not(target_os = "linux"))]
	{
		let _ = (app, dx, dy, persist);
	}
}

/// Maps the overlay at its recording size (380x64 from the window config).
/// Positioning: layer-shell margins on Wayland (see `overlay_layer`), the
/// saved position on macOS, the window manager elsewhere.
fn show_overlay(app: &tauri::AppHandle) {
	let Some(window) = app.get_webview_window("overlay") else {
		return;
	};
	let _ = window.show();
}

/// Unmaps the overlay; on error keeps it around briefly to show the message.
fn hide_overlay(app: &tauri::AppHandle, failed: bool) {
	let Some(window) = app.get_webview_window("overlay") else {
		return;
	};
	if !failed {
		let _ = window.hide();
		return;
	}
	std::thread::sleep(std::time::Duration::from_millis(2500));
	let _ = window.hide();
}

/// Emits `level` events (~10 Hz) while a recorder is active.
fn spawn_level_ticker(app: tauri::AppHandle) {
	std::thread::spawn(move || loop {
		std::thread::sleep(std::time::Duration::from_millis(100));
		let state = app.state::<Dictation>();
		let guard = state.recorder.lock().unwrap();
		let Some(recorder) = guard.as_ref() else {
			break;
		};
		let _ = app.emit("level", recorder.level());
	});
}

fn transcribe_and_paste(app: &tauri::AppHandle, samples: &[f32]) -> Result<(), String> {
	let settings = settings::load();
	let language = match settings.language.as_str() {
		"auto" => None,
		code => Some(code),
	};
	let prompt = settings
		.active_prompt
		.as_ref()
		.and_then(|name| settings.prompts.iter().find(|p| &p.name == name))
		.map(|p| p.text.as_str());
	let (text, lang) = match settings.engine {
		settings::Engine::Gigaam => {
			match models::resolve_gigaam(settings.gigaam_model_id.as_deref()) {
				Some(model) => (
					transcribe_gigaam_cached(app, &model, samples)?,
					"ru".to_string(),
				),
				None => {
					// No GigaAM model: whisper handles Russian, and the user gets
					// one nudge per session towards the better engine.
					suggest_gigaam_download(app);
					let model = require_whisper_model(app)?;
					transcribe_whisper_cached(app, &model, samples, Some("ru"), prompt)?
				}
			}
		}
		settings::Engine::Qwen => {
			let model = models::resolve_qwen(settings.qwen_model_id.as_deref())
				.ok_or_else(|| no_model(app, "no Qwen3-ASR model available: download one in settings"))?;
			transcribe_qwen_cached(app, &model, samples, language)?
		}
		settings::Engine::Whisper => {
			let model = require_whisper_model(app)?;
			transcribe_whisper_cached(app, &model, samples, language, prompt)?
		}
	};
	// History is written before pasting: a paste failure must not lose the
	// transcript. Its error is deferred so a history failure does not block
	// the paste either.
	let history = history::open().and_then(|conn| history::insert(&conn, &text, &lang));
	// The `transcript` event is what refreshes the History tab; it must not
	// be gated on paste success — a failed paste still leaves the new row in
	// the database, and an open History window would otherwise stay stale.
	let _ = app.emit("transcript", &text);
	inject::paste(&text)?;
	history?;
	Ok(())
}

/// Resolves the whisper model or opens the settings window on the model
/// picker instead of failing silently (first run).
fn require_whisper_model(app: &tauri::AppHandle) -> Result<PathBuf, String> {
	let settings = settings::load();
	models::resolve(settings.model_path.as_deref(), settings.model_id.as_deref())
		.ok_or_else(|| no_model(app, "no model available: download one in settings"))
}

/// Surfaces the settings window for a missing model and returns `message`
/// as the pipeline error.
fn no_model(app: &tauri::AppHandle, message: &str) -> String {
	if let Some(window) = app.get_webview_window("main") {
		let _ = window.show();
		let _ = window.set_focus();
	}
	let _ = app.emit("no-model", ());
	message.to_string()
}

const GIGAAM_SUGGESTION: &str =
	"Russian dictation can use the GigaAM engine - download it in Settings, Model section.";

/// One-time-per-session desktop notification: Russian dictation could use
/// the GigaAM engine (DESIGN.md "Engines"). Fire-and-forget: portal
/// notification failures only cost the suggestion.
fn suggest_gigaam_download(app: &tauri::AppHandle) {
	let state = app.state::<Dictation>();
	if state.gigaam_suggested.swap(true, Ordering::Relaxed) {
		return;
	}
	let _ = app.emit("gigaam-suggest", ());
	#[cfg(target_os = "macos")]
	let _ = std::process::Command::new("osascript")
		.args(["-e", &format!("display notification \"{GIGAAM_SUGGESTION}\" with title \"wtf\"")])
		.status();
	#[cfg(target_os = "linux")]
	tauri::async_runtime::spawn(async move {
		use ashpd::desktop::notification::NotificationProxy;
		let Ok(proxy) = NotificationProxy::new().await else {
			return;
		};
		let notification = ashpd::desktop::notification::Notification::new("wtf")
			.body(GIGAAM_SUGGESTION);
		let _ = proxy.add_notification("wtf-gigaam-suggest", notification).await;
	});
}

/// Drops all cached engines: model weights and whisper state are freed
/// (GPU VRAM + host memory); the next dictation reloads lazily.
pub fn unload_transcriber(app: &tauri::AppHandle) {
	let state = app.state::<Dictation>();
	let whisper = state.whisper.lock().unwrap().take().is_some();
	let gigaam = state.gigaam.lock().unwrap().take().is_some();
	let qwen = state.qwen.lock().unwrap().take().is_some();
	if whisper || gigaam || qwen {
		eprintln!("models unloaded");
	}
}

/// Drops only the cached Qwen3-ASR engine (its model files changed or were
/// deleted).
pub fn unload_qwen(app: &tauri::AppHandle) {
	let state = app.state::<Dictation>();
	if state.qwen.lock().unwrap().take().is_some() {
		eprintln!("qwen model unloaded");
	}
}

/// Drops only the cached GigaAM engine (its model file changed or was
/// deleted).
pub fn unload_gigaam(app: &tauri::AppHandle) {
	let state = app.state::<Dictation>();
	if state.gigaam.lock().unwrap().take().is_some() {
		eprintln!("gigaam model unloaded");
	}
}

#[tauri::command]
pub fn unload_model(app: tauri::AppHandle) {
	unload_transcriber(&app);
}

/// An engine slot of `Dictation`: the loaded engine and its model path.
type Slot = Option<(PathBuf, asr::Transcriber)>;

/// Loads the whisper model into `slot` unless it already holds that model.
fn load_whisper(slot: &mut Slot, model: &std::path::Path) -> Result<(), String> {
	if slot.as_ref().is_some_and(|(path, _)| path == model) {
		return Ok(());
	}
	let settings = settings::load();
	eprintln!("loading model {}...", model.display());
	*slot = Some((
		model.to_path_buf(),
		asr::Transcriber::whisper(
			&model.to_string_lossy(),
			settings.gpu_device,
			settings.use_gpu,
		)?,
	));
	Ok(())
}

/// Loads the GigaAM model into `slot` unless it already holds that model.
fn load_gigaam(slot: &mut Slot, model: &std::path::Path) -> Result<(), String> {
	if slot.as_ref().is_some_and(|(path, _)| path == model) {
		return Ok(());
	}
	eprintln!("loading gigaam model {}...", model.display());
	*slot = Some((
		model.to_path_buf(),
		asr::Transcriber::GigaAm(crate::gigaam::GigaAm::new(model)?),
	));
	Ok(())
}

/// Starts the Qwen3-ASR server for `model` in `slot` unless it already runs
/// that model.
fn load_qwen(slot: &mut Slot, model: &std::path::Path) -> Result<(), String> {
	if slot.as_ref().is_some_and(|(path, _)| path == model) {
		return Ok(());
	}
	let settings = settings::load();
	eprintln!("loading qwen model {}...", model.display());
	*slot = Some((
		model.to_path_buf(),
		asr::Transcriber::Qwen(crate::qwen::Qwen::new(
			model,
			settings.qwen_use_gpu,
			settings.qwen_gpu_device.as_deref(),
			settings.qwen_context,
		)?),
	));
	Ok(())
}

/// Loads the engine a dictation would use right now (same routing as
/// `transcribe_and_paste`), so the first dictation does not wait for it.
/// A missing model is left for the first dictation to report.
pub fn preload(app: &tauri::AppHandle) -> Result<(), String> {
	let settings = settings::load();
	let state = app.state::<Dictation>();
	let whisper = || models::resolve(settings.model_path.as_deref(), settings.model_id.as_deref());
	match settings.engine {
		settings::Engine::Gigaam => {
			if let Some(model) = models::resolve_gigaam(settings.gigaam_model_id.as_deref()) {
				load_gigaam(&mut state.gigaam.lock().unwrap(), &model)?;
			} else if let Some(model) = whisper() {
				load_whisper(&mut state.whisper.lock().unwrap(), &model)?;
			}
		}
		settings::Engine::Qwen => {
			if let Some(model) = models::resolve_qwen(settings.qwen_model_id.as_deref()) {
				load_qwen(&mut state.qwen.lock().unwrap(), &model)?;
			}
		}
		settings::Engine::Whisper => {
			if let Some(model) = whisper() {
				load_whisper(&mut state.whisper.lock().unwrap(), &model)?;
			}
		}
	}
	Ok(())
}

fn transcribe_whisper_cached(
	app: &tauri::AppHandle,
	model: &std::path::Path,
	samples: &[f32],
	language: Option<&str>,
	initial_prompt: Option<&str>,
) -> Result<(String, String), String> {
	let state = app.state::<Dictation>();
	let mut cached = state.whisper.lock().unwrap();
	load_whisper(&mut cached, model)?;
	cached
		.as_mut()
		.expect("transcriber was just stored")
		.1
		.transcribe(samples, language, initial_prompt)
}

fn transcribe_gigaam_cached(
	app: &tauri::AppHandle,
	model: &std::path::Path,
	samples: &[f32],
) -> Result<String, String> {
	let state = app.state::<Dictation>();
	let mut cached = state.gigaam.lock().unwrap();
	load_gigaam(&mut cached, model)?;
	let (text, _lang) = cached
		.as_mut()
		.expect("transcriber was just stored")
		.1
		.transcribe(samples, Some("ru"), None)?;
	Ok(text)
}

fn transcribe_qwen_cached(
	app: &tauri::AppHandle,
	model: &std::path::Path,
	samples: &[f32],
	language: Option<&str>,
) -> Result<(String, String), String> {
	let state = app.state::<Dictation>();
	let mut cached = state.qwen.lock().unwrap();
	load_qwen(&mut cached, model)?;
	cached
		.as_mut()
		.expect("transcriber was just stored")
		.1
		.transcribe(samples, language, None)
}
