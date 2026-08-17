//! Global recording hotkey via the xdg-desktop-portal GlobalShortcuts
//! interface (ashpd, feature `global_shortcuts`).
//!
//! Design (DESIGN.md, "Pipeline"): press-to-toggle record; a second global
//! shortcut cycles the dictation language. Plasma 6 shows a native binding
//! dialog on first registration.

use ashpd::desktop::global_shortcuts::{GlobalShortcuts, NewShortcut};
use futures_util::StreamExt;

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

/// Calls `on_activation(shortcut_id)` for every shortcut activation until the
/// session ends. Runs forever on success; the portal session stays alive as
/// long as `globals` is kept alive by the caller.
pub async fn listen<F>(globals: &GlobalShortcuts, mut on_activation: F) -> Result<(), ashpd::Error>
where
	F: FnMut(&str),
{
	let mut activations = globals.receive_activated().await?;
	while let Some(activated) = activations.next().await {
		on_activation(activated.shortcut_id());
	}
	Ok(())
}
