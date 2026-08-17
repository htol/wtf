<script lang="ts">
	import { listen } from '@tauri-apps/api/event';
	import { onMount } from 'svelte';
	import Settings from './lib/Settings.svelte';
	import History from './lib/History.svelte';

	type Tab = 'settings' | 'history';
	let tab: Tab = $state('settings');
	// Step-1 wiring check: `record` toggles the indicator (press-to-toggle,
	// DESIGN.md "Pipeline"); other shortcuts show as the last action.
	let recording = $state(false);
	let lastAction = $state<string | null>(null);

	onMount(() => {
		const unlisten = listen<string>('shortcut', (event) => {
			const at = new Date().toLocaleTimeString();
			if (event.payload === 'record') {
				recording = !recording;
				lastAction = `record ${recording ? 'started' : 'stopped'} at ${at}`;
			} else {
				lastAction = `${event.payload} at ${at}`;
			}
		});
		return () => {
			unlisten.then((u) => u());
		};
	});
</script>

<div class="root">
	<nav>
		<button class:active={tab === 'settings'} onclick={() => (tab = 'settings')}>Settings</button>
		<button class:active={tab === 'history'} onclick={() => (tab = 'history')}>History</button>
	</nav>
	<main>
		{#if tab === 'settings'}
			<Settings />
		{:else}
			<History />
		{/if}
	</main>
	<footer>
		{#if recording}
			<span class="accent">● recording</span> — press the hotkey again to stop
		{:else if lastAction}
			{lastAction}
		{:else}
			Hotkeys armed — press a global shortcut to test
		{/if}
	</footer>
</div>

<style>
	.root {
		display: flex;
		flex-direction: column;
		height: 100%;
	}

	nav {
		display: flex;
		gap: 4px;
		padding: 8px 12px 0;
		background: var(--nord1);
	}

	nav button {
		border: none;
		border-radius: 4px 4px 0 0;
		background: transparent;
		color: var(--nord4);
		padding: 8px 18px;
	}

	nav button.active {
		background: var(--nord0);
		color: var(--nord8);
	}

	main {
		flex: 1;
		overflow-y: auto;
		padding: 16px;
	}

	footer {
		padding: 6px 12px;
		background: var(--nord1);
		color: var(--nord3);
		font-size: 12px;
	}

	footer .accent {
		color: var(--nord8);
	}
</style>
