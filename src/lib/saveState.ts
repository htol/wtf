// Bridge between the save-owning tabs (Settings, Prompts) and the Save
// button in the main window's tab bar: the form state lives in the tab
// component, the button lives in App.svelte's title bar. The active tab
// registers its save callback and dirty flag here; App renders the button
// only on tabs that registered.
import { writable } from 'svelte/store';

export interface SaveBar {
	/** Unsaved changes in the Settings form. */
	dirty: boolean;
	/** Time of the last successful save, shown next to the button. */
	savedAt: string | null;
	/** Save callback registered by the owning tab; null while it loads. */
	save: (() => void) | null;
}

export const saveBar = writable<SaveBar>({ dirty: false, savedAt: null, save: null });

/**
 * Release the bar on unmount, but only if no other tab has already taken
 * it over — tab switches destroy the old component and mount the new one
 * in one commit, and the order must not matter.
 */
export function releaseSaveBar(owner: () => void) {
	saveBar.update((s) =>
		s.save === owner ? { dirty: false, savedAt: null, save: null } : s
	);
}
