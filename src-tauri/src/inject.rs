//! Text injection: clipboard + simulated Ctrl+V, with clipboard restore.
//!
//! Design (DESIGN.md, "Pipeline"): paste everywhere Ctrl+V works. Uses
//! wl-copy/wl-paste for the clipboard and ydotool for the keypress.
//! `ydotoold` must be running.

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Duration;

/// Restoring races with clipboard managers; a short settle delay mitigates it.
const RESTORE_DELAY: Duration = Duration::from_millis(120);

fn run(args: &[&str]) -> Result<String, String> {
	let out = Command::new(args[0])
		.args(&args[1..])
		.output()
		.map_err(|e| format!("{}: {e}", args[0]))?;
	if out.status.success() {
		Ok(String::from_utf8_lossy(&out.stdout).into_owned())
	} else {
		Err(format!(
			"{} failed: {}",
			args[0],
			String::from_utf8_lossy(&out.stderr)
		))
	}
}

fn set_clipboard(text: &str) -> Result<(), String> {
	let mut child = Command::new("wl-copy")
		.stdin(Stdio::piped())
		.spawn()
		.map_err(|e| format!("wl-copy: {e}"))?;
	if let Some(stdin) = child.stdin.as_mut() {
		stdin
			.write_all(text.as_bytes())
			.map_err(|e| format!("wl-copy stdin: {e}"))?;
	} else {
		return Err("wl-copy has no stdin".into());
	}
	child.wait().map_err(|e| e.to_string())?;
	Ok(())
}

fn press_paste() -> Result<(), String> {
	// evdev keycodes: KEY_LEFTSHIFT=42, KEY_INSERT=110; 1 = press, 0 = release.
	// Shift+Insert instead of Ctrl+V: V translates to a Cyrillic keysym under
	// a Russian layout (Ctrl+М), which apps' Ctrl+V bindings never match;
	// Shift and Insert are layout-invariant and paste in terminals, GTK, Qt.
	run(&["ydotool", "key", "42:1", "110:1", "110:0", "42:0"])?;
	Ok(())
}

/// Puts `text` into the focused application via clipboard + Ctrl+V, then
/// restores the previous clipboard content (if any).
pub fn paste(text: &str) -> Result<(), String> {
	let saved = run(&["wl-paste", "--no-newline"]).ok();
	set_clipboard(text)?;
	press_paste()?;;
	std::thread::sleep(RESTORE_DELAY);
	if let Some(previous) = saved {
		set_clipboard(&previous)?;
	}
	Ok(())
}

/// Copies `text` to the clipboard (wl-copy, same path as paste uses).
#[tauri::command]
pub fn copy_to_clipboard(text: String) -> Result<(), String> {
	set_clipboard(&text)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn set_clipboard_roundtrip() {
		// The test drives the real clipboard; save and restore the user's
		// current content so a test run doesn't clobber it.
		let saved = run(&["wl-paste", "--no-newline"]).ok();
		set_clipboard("wtf-test-marker").unwrap();
		let got = run(&["wl-paste", "--no-newline"]).unwrap();
		assert_eq!(got, "wtf-test-marker");
		if let Some(previous) = saved {
			set_clipboard(&previous).unwrap();
		}
	}
}
