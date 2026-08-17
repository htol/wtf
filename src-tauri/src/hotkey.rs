//! Global recording hotkey via the xdg-desktop-portal GlobalShortcuts
//! interface (ashpd, feature `global_shortcuts`).
//!
//! Design (DESIGN.md, "Pipeline"): press-to-toggle record; a second global
//! shortcut cycles the dictation language. Plasma 6 shows a native binding
//! dialog on first registration.
//!
//! TODO: after `register()`, spawn a task looping over
//! `globals.receive_activated()` and forward each `Activated::shortcut_id()`
//! into the app (toggle recording / cycle language).

use ashpd::desktop::global_shortcuts::{GlobalShortcuts, NewShortcut};

pub const SHORTCUT_RECORD: &str = "record";
pub const SHORTCUT_CYCLE_LANGUAGE: &str = "cycle-language";

/// Creates the portal session and binds both shortcuts.
/// Returns the proxy to listen on for activations.
pub async fn register() -> Result<GlobalShortcuts, ashpd::Error> {
	let globals = GlobalShortcuts::new().await?;
	let session = globals.create_session(Default::default()).await?;
	let request = globals
		.bind_shortcuts(
			&session,
			&[
				NewShortcut::new(SHORTCUT_RECORD, "Toggle recording"),
				NewShortcut::new(SHORTCUT_CYCLE_LANGUAGE, "Cycle language"),
			],
			None,
			Default::default(),
		)
		.await?;
	request.response()?;
	Ok(globals)
}
