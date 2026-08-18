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

/// The portal session plus its proxy. Kept in app state for the daemon's
/// lifetime: dropping the session closes it and unbinds the shortcuts;
/// keeping it lets `rebind_shortcuts` reopen the Plasma binding dialog on
/// demand.
pub struct Hotkeys {
	pub globals: GlobalShortcuts,
	session: ashpd::desktop::Session<GlobalShortcuts>,
}

fn shortcut_specs() -> Vec<NewShortcut> {
	vec![
		NewShortcut::new(SHORTCUT_RECORD, "Toggle recording"),
		NewShortcut::new(SHORTCUT_CYCLE_LANGUAGE, "Cycle language"),
	]
}

/// Creates the portal session and binds both shortcuts (the Plasma dialog
/// shows up for any shortcut not yet bound by the user).
pub async fn register() -> Result<Hotkeys, ashpd::Error> {
	let globals = GlobalShortcuts::new().await?;
	let session = globals.create_session(Default::default()).await?;
	let request = globals
		.bind_shortcuts(&session, &shortcut_specs(), None, Default::default())
		.await?;
	request.response()?;
	Ok(Hotkeys { globals, session })
}

/// Re-runs the binding dialog for the existing session (Settings tab).
#[tauri::command]
pub async fn rebind_shortcuts(app: tauri::AppHandle) -> Result<(), String> {
	use tauri::Manager;

	let hub = app
		.state::<std::sync::Arc<Hotkeys>>()
		.inner()
		.clone();
	let request = hub
		.globals
		.bind_shortcuts(&hub.session, &shortcut_specs(), None, Default::default())
		.await
		.map_err(|e| format!("portal rebinding failed: {e}"))?;
	request
		.response()
		.map(|_| ())
		.map_err(|e| format!("shortcut binding cancelled: {e}"))
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
