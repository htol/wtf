<script lang="ts">
	import { listen } from '@tauri-apps/api/event';
	import { onMount } from 'svelte';
	import Settings from './lib/Settings.svelte';
	import History from './lib/History.svelte';
	import Prompts from './lib/Prompts.svelte';

	type Tab = 'history' | 'prompts' | 'settings';
	let tab: Tab = $state('history');
	// Recording state is owned by the backend pipeline; the footer mirrors
	// it from `recording` events.
	let recording = $state(false);
	let processing = $state(false);
	let lastAction = $state<string | null>(null);
	let pipelineError = $state<string | null>(null);

	onMount(() => {
		const unlistenShortcut = listen<string>('shortcut', (event) => {
			const at = new Date().toLocaleTimeString();
			lastAction = `${event.payload} at ${at}`;
		});
		const unlistenRecording = listen<boolean>('recording', (event) => {
			recording = event.payload;
		});
		const unlistenNoModel = listen('no-model', () => {
			tab = 'settings';
		});
		const unlistenError = listen<string>('pipeline-error', (event) => {
			pipelineError = event.payload;
		});
		const unlistenProcessing = listen<boolean>('processing', (event) => {
			processing = event.payload;
		});
		return () => {
			unlistenShortcut.then((u) => u());
			unlistenRecording.then((u) => u());
			unlistenNoModel.then((u) => u());
			unlistenError.then((u) => u());
			unlistenProcessing.then((u) => u());
		};
	});
</script>

<div class="root">
	<nav>
		<button class:active={tab === 'history'} onclick={() => (tab = 'history')}>History</button>
		<button class:active={tab === 'prompts'} onclick={() => (tab = 'prompts')}>Prompts</button>
		<button class:active={tab === 'settings'} onclick={() => (tab = 'settings')}>Settings</button>
	</nav>
	<main>
		{#if tab === 'history'}
			<History />
		{:else if tab === 'prompts'}
			<Prompts />
		{:else}
			<Settings />
		{/if}
	</main>
	<footer>
		{#if recording}
			<span class="accent">● recording</span> — press the hotkey again to stop
		{:else if processing}
			<span class="accent">◌ transcribing…</span>
		{:else if pipelineError}
			<span class="error">{pipelineError}</span>
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

	footer .error {
		color: var(--nord11);
	}
</style>
