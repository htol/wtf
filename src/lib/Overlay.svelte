<script lang="ts">
	import { invoke } from '@tauri-apps/api/core';
	import { currentMonitor, getCurrentWindow } from '@tauri-apps/api/window';
	import { listen } from '@tauri-apps/api/event';
	import { onMount } from 'svelte';

	type Status = 'idle' | 'recording' | 'processing' | 'error';

	let status = $state<Status>('recording');
	let level = $state(0);
	let language = $state('auto');
	let errorMessage = $state<string | null>(null);
	// Collapsed = 1x1 invisible idle state (the window stays mapped; see
	// pipeline::prime_overlay). Expanded = recording pill.
	let collapsed = $state(true);

	async function refreshLanguage() {
		const settings = await invoke<{ language: string }>('get_settings');
		language = settings.language;
	}

	// Report the window position (as fractions of the monitor) so the next
	// session can restore it. Debounced: drag events fire continuously.
	let saveTimer: ReturnType<typeof setTimeout> | null = null;
	function scheduleSave() {
		if (saveTimer) clearTimeout(saveTimer);
		saveTimer = setTimeout(async () => {
			const win = getCurrentWindow();
			const [position, monitor] = await Promise.all([win.outerPosition(), currentMonitor()]);
			if (!monitor) return;
			const x = position.x / Math.max(1, monitor.size.width);
			const y = position.y / Math.max(1, monitor.size.height);
			await invoke('set_overlay_position', {
				x: Math.min(1, Math.max(0, x)),
				y: Math.min(1, Math.max(0, y))
			});
		}, 400);
	}

	onMount(() => {
		refreshLanguage();
		const win = getCurrentWindow();
		const syncCollapsed = (size: { width: number }) => {
			collapsed = size.width < 100;
		};
		win.innerSize().then(syncCollapsed);
		const unlistenResized = win.onResized((event) => syncCollapsed(event.payload));
		const unlistenRecording = listen<boolean>('recording', (event) => {
			if (event.payload) {
				status = 'recording';
				errorMessage = null;
				refreshLanguage();
			} else if (status === 'recording') {
				status = 'idle';
			}
		});
		const unlistenProcessing = listen<boolean>('processing', (event) => {
			status = event.payload ? 'processing' : 'idle';
		});
		const unlistenLevel = listen<number>('level', (event) => {
			level = event.payload;
		});
		const unlistenError = listen<string>('pipeline-error', (event) => {
			status = 'error';
			errorMessage = event.payload;
		});
		const unlistenMoved = getCurrentWindow().onMoved(() => scheduleSave());
		return () => {
			unlistenRecording.then((u) => u());
			unlistenProcessing.then((u) => u());
			unlistenLevel.then((u) => u());
			unlistenError.then((u) => u());
			unlistenMoved.then((u) => u());
			unlistenResized.then((u) => u());
		};
	});
</script>

<div class="overlay" class:collapsed data-tauri-drag-region>
	{#if status === 'recording'}
		<span class="dot recording" data-tauri-drag-region></span>
	{:else if status === 'processing'}
		<span class="dot processing" data-tauri-drag-region></span>
	{:else if status === 'error'}
		<span class="dot error" data-tauri-drag-region></span>
	{/if}

	{#if status === 'error' && errorMessage}
		<span class="text error-text" data-tauri-drag-region>{errorMessage}</span>
	{:else if status !== 'idle'}
		<span class="text" data-tauri-drag-region>
			{status === 'recording' ? 'recording' : 'transcribing'}
		</span>
		{#if status === 'recording'}
			<span class="meter" data-tauri-drag-region>
				<span class="fill" style="width: {Math.min(100, Math.pow(Math.min(1, level * 2), 0.25) * 100)}%"></span>
			</span>
		{/if}
		<span class="lang" data-tauri-drag-region>{language}</span>
	{/if}
</div>

<style>
	:global(html),
	:global(body) {
		background: transparent;
		height: 100%;
		margin: 0;
		overflow: hidden;
	}

	:global(#app) {
		height: 100%;
	}

	.overlay {
		display: flex;
		align-items: center;
		gap: 8px;
		height: 100%;
		padding: 0 10px;
		background: rgba(46, 52, 64, 0.92);
		border: 1px solid var(--nord3);
		border-radius: 10px;
		cursor: grab;
		user-select: none;
	}

	.overlay:active {
		cursor: grabbing;
	}

	.overlay.collapsed {
		background: transparent;
		border: none;
	}

	.dot {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		flex-shrink: 0;
	}

	.dot.recording {
		background: var(--nord11);
		animation: pulse 1.2s ease-in-out infinite;
	}

	.dot.processing {
		background: transparent;
		border: 2px solid var(--nord8);
		border-top-color: transparent;
		animation: spin 0.9s linear infinite;
	}

	.dot.error {
		background: var(--nord11);
	}

	.text {
		font-size: 11px;
		color: var(--nord4);
		white-space: nowrap;
	}

	.error-text {
		color: var(--nord11);
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}

	.meter {
		flex: 1;
		height: 4px;
		min-width: 40px;
		border-radius: 2px;
		background: var(--nord1);
		overflow: hidden;
	}

	.fill {
		display: block;
		height: 100%;
		border-radius: 2px;
		background: var(--nord8);
		transition: width 100ms linear;
	}

	.lang {
		font-size: 10px;
		text-transform: uppercase;
		letter-spacing: 0.06em;
		color: var(--nord7);
		flex-shrink: 0;
	}

	@keyframes pulse {
		0%,
		100% {
			opacity: 1;
		}
		50% {
			opacity: 0.35;
		}
	}

	@keyframes spin {
		to {
			transform: rotate(360deg);
		}
	}
</style>
