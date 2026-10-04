//! Text injection on macOS: clipboard + simulated Cmd+V, with clipboard
//! restore.
//!
//! Uses pbcopy/pbpaste for the clipboard and a Quartz keyboard event for
//! the keypress. Posting the event needs the Accessibility permission
//! (System Settings, Privacy & Security).

use std::ffi::c_void;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Duration;

/// Restoring races with the target app reading the clipboard; a short
/// settle delay mitigates it.
const RESTORE_DELAY: Duration = Duration::from_millis(120);

/// kVK_ANSI_V: the physical V key, whatever the active layout.
const KEY_V: u16 = 9;
/// kCGEventFlagMaskCommand.
const FLAG_COMMAND: u64 = 1 << 20;
/// kCGHIDEventTap.
const HID_EVENT_TAP: u32 = 0;

#[link(name = "ApplicationServices", kind = "framework")]
extern "C" {
	fn AXIsProcessTrusted() -> u8;
	fn CGEventCreateKeyboardEvent(source: *const c_void, key: u16, key_down: bool) -> *mut c_void;
	fn CGEventSetFlags(event: *mut c_void, flags: u64);
	fn CGEventPost(tap: u32, event: *mut c_void);
}

#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
	fn CFRelease(object: *const c_void);
}

/// pbcopy/pbpaste pick the text encoding from the locale, and an app
/// started from Finder has none: without this, non-ASCII text is mangled.
fn clipboard_tool(name: &str) -> Command {
	let mut command = Command::new(name);
	command.env("LC_CTYPE", "UTF-8");
	command
}

fn get_clipboard() -> Result<String, String> {
	let out = clipboard_tool("pbpaste")
		.output()
		.map_err(|e| format!("pbpaste: {e}"))?;
	if out.status.success() {
		Ok(String::from_utf8_lossy(&out.stdout).into_owned())
	} else {
		Err(format!("pbpaste failed: {}", String::from_utf8_lossy(&out.stderr)))
	}
}

fn set_clipboard(text: &str) -> Result<(), String> {
	let mut child = clipboard_tool("pbcopy")
		.stdin(Stdio::piped())
		.spawn()
		.map_err(|e| format!("pbcopy: {e}"))?;
	// Taking stdin closes it after the write: pbcopy reads until EOF.
	let Some(mut stdin) = child.stdin.take() else {
		return Err("pbcopy has no stdin".into());
	};
	stdin
		.write_all(text.as_bytes())
		.map_err(|e| format!("pbcopy stdin: {e}"))?;
	drop(stdin);
	child.wait().map_err(|e| e.to_string())?;
	Ok(())
}

fn press_paste() -> Result<(), String> {
	// Without the permission the events are dropped silently.
	if unsafe { AXIsProcessTrusted() } == 0 {
		return Err(
			"no Accessibility permission: allow wtf in System Settings, Privacy & Security, Accessibility"
				.into(),
		);
	}
	for key_down in [true, false] {
		// A null source means the default event source.
		let event = unsafe { CGEventCreateKeyboardEvent(std::ptr::null(), KEY_V, key_down) };
		if event.is_null() {
			return Err("cannot create a keyboard event".into());
		}
		unsafe {
			CGEventSetFlags(event, FLAG_COMMAND);
			CGEventPost(HID_EVENT_TAP, event);
			CFRelease(event);
		}
	}
	Ok(())
}

/// Puts `text` into the focused application via clipboard + Cmd+V, then
/// restores the previous clipboard content (if any).
pub fn paste(text: &str) -> Result<(), String> {
	let saved = get_clipboard().ok();
	set_clipboard(text)?;
	press_paste()?;
	std::thread::sleep(RESTORE_DELAY);
	if let Some(previous) = saved {
		set_clipboard(&previous)?;
	}
	Ok(())
}

/// Copies `text` to the clipboard (pbcopy, same path as paste uses).
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
		let saved = get_clipboard().ok();
		set_clipboard("wtf-test-marker привет").unwrap();
		let got = get_clipboard().unwrap();
		assert_eq!(got, "wtf-test-marker привет");
		if let Some(previous) = saved {
			set_clipboard(&previous).unwrap();
		}
	}
}
