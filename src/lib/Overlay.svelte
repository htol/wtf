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
	// Linux/Wayland: the overlay is a layer-shell surface, which the
	// compositor cannot move, so it moves itself (see pipeline::shift_overlay).
	let layer = $state(false);

	// Live waveform: bar heights (0..1) eased towards a target shaped by the
	// microphone RMS (`level`, ~10 Hz) and a per-bar wobble, so the bars keep
	// moving between level updates instead of stepping.
	const BARS = 18;
	const phases = Array.from({ length: BARS }, () => Math.random() * Math.PI * 2);
	let bars = $state(Array.from({ length: BARS }, () => 0.12));
	function animate(now: number) {
		const amp = Math.min(1, Math.sqrt(Math.min(1, level * 6)));
		const t = now / 1000;
		bars = bars.map((current, i) => {
			const mid = (BARS - 1) / 2;
			const envelope = 1 - Math.abs(i - mid) / mid;
			const wobble = 0.55 + 0.45 * Math.sin(t * 9 + phases[i]);
			const target = 0.12 + amp * (0.3 + 0.7 * envelope) * wobble;
			const ease = target > current ? 0.45 : 0.18;
			return current + (target - current) * ease;
		});
	}
	$effect(() => {
		if (status !== 'recording') return;
		let frame = requestAnimationFrame(function tick(now) {
			animate(now);
			frame = requestAnimationFrame(tick);
		});
		return () => cancelAnimationFrame(frame);
	});

	const label = $derived(
		status === 'recording'
			? 'Recording'
			: status === 'processing'
				? 'Transcribing…'
				: status === 'error'
					? (errorMessage ?? 'Error')
					: ''
	);

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

	// Drag of the layer surface. Each step sends the pointer's offset from
	// where the drag started, in the window's own coordinates: once the
	// surface has moved under the pointer the offset is back to zero, so a
	// late or dropped step corrects itself instead of accumulating. One step
	// at a time, and the next only after a frame has passed since the step
	// was applied — the compositor moves the surface on the next commit, and
	// a step sent before that would be measured against the old position.
	let drag: { x: number; y: number } | null = null;
	let dragBusy = false;
	function onPointerDown(event: PointerEvent) {
		if (!layer || event.button !== 0) return;
		drag = { x: event.clientX, y: event.clientY };
		(event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
	}
	async function onPointerMove(event: PointerEvent) {
		if (!drag || dragBusy) return;
		const dx = Math.round(event.clientX - drag.x);
		const dy = Math.round(event.clientY - drag.y);
		if (dx === 0 && dy === 0) return;
		dragBusy = true;
		try {
			await invoke('shift_overlay', { dx, dy, persist: false });
			await new Promise(requestAnimationFrame);
		} finally {
			dragBusy = false;
		}
	}
	function onPointerUp() {
		if (!drag) return;
		drag = null;
		invoke('shift_overlay', { dx: 0, dy: 0, persist: true });
	}
	// Tauri's drag-region script listens on document mousedown and would ask
	// the compositor to move the window; a layer surface has no such move.
	function onMouseDown(event: MouseEvent) {
		if (layer) event.stopPropagation();
	}

	onMount(() => {
		refreshLanguage();
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
		// A layer surface reports no position: its moves are saved by
		// shift_overlay, and a moved event would only store zeros.
		const unlistenMoved = invoke<boolean>('overlay_is_layer').then((isLayer) => {
			layer = isLayer;
			return isLayer ? () => {} : getCurrentWindow().onMoved(() => scheduleSave());
		});
		return () => {
			unlistenRecording.then((u) => u());
			unlistenProcessing.then((u) => u());
			unlistenLevel.then((u) => u());
			unlistenError.then((u) => u());
			unlistenMoved.then((u) => u());
		};
	});
</script>

<!-- svelte-ignore a11y_no_static_element_interactions -- a drag handle on a
	window that never takes keyboard focus -->
<div
	class="frame"
	data-tauri-drag-region
	onmousedown={onMouseDown}
	onpointerdown={onPointerDown}
	onpointermove={onPointerMove}
	onpointerup={onPointerUp}
	onpointercancel={onPointerUp}
>
	<div class="pill" class:error={status === 'error'} data-tauri-drag-region>
		<div class="glyph" data-tauri-drag-region>
			{#if status === 'recording'}
				<div class="bars" data-tauri-drag-region>
					{#each bars as height, i (i)}
						<span class="bar" style="transform: scaleY({height})"></span>
					{/each}
				</div>
			{:else if status === 'processing'}
				<div class="bars busy" data-tauri-drag-region>
					{#each phases as _, i (i)}
						<span class="bar" style="animation-delay: {i * 55}ms"></span>
					{/each}
				</div>
			{:else if status === 'error'}
				<span class="alert" data-tauri-drag-region>!</span>
			{/if}
		</div>
		<span class="text" class:error-text={status === 'error'} data-tauri-drag-region>{label}</span>
		{#if status !== 'error'}
			<span class="lang" data-tauri-drag-region>{language}</span>
		{/if}
	</div>
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

	/* The window is 380x64; the frame keeps 8px clear around the pill for
	   its shadow (the window has no compositor shadow of its own). */
	.frame {
		box-sizing: border-box;
		display: flex;
		height: 100%;
		padding: 8px;
		cursor: grab;
		user-select: none;
		touch-action: none;
	}

	.frame:active {
		cursor: grabbing;
	}

	/* Hugs its content; only an error message may stretch to the window. */
	.pill {
		flex: 0 1 auto;
		max-width: 100%;
		display: flex;
		align-items: center;
		gap: 12px;
		min-width: 0;
		padding: 0 14px 0 16px;
		background: rgba(46, 52, 64, 0.94);
		border: 1px solid rgba(216, 222, 233, 0.14);
		border-radius: 999px;
		box-shadow:
			0 6px 24px rgba(0, 0, 0, 0.45),
			inset 0 1px 0 rgba(236, 239, 244, 0.06);
	}

	.pill.error {
		border-color: rgba(191, 97, 106, 0.6);
	}

	.glyph {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 90px;
		height: 28px;
		flex-shrink: 0;
	}

	.bars {
		display: flex;
		align-items: center;
		gap: 2px;
		height: 100%;
	}

	.bar {
		display: block;
		width: 3px;
		height: 100%;
		border-radius: 2px;
		background: var(--nord8);
		transform: scaleY(0.12);
		transform-origin: center;
	}

	.bars.busy .bar {
		background: var(--nord9);
		animation: sweep 1.1s ease-in-out infinite;
	}

	.alert {
		display: grid;
		place-items: center;
		width: 22px;
		height: 22px;
		border-radius: 50%;
		background: var(--nord11);
		color: var(--nord6);
		font-size: 14px;
		font-weight: 700;
	}

	.text {
		flex: 0 1 auto;
		min-width: 0;
		font-size: 14px;
		font-weight: 500;
		line-height: 1.3;
		letter-spacing: 0.01em;
		color: var(--nord6);
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}

	.error-text {
		color: var(--nord11);
	}

	.lang {
		flex-shrink: 0;
		padding: 4px 9px;
		border-radius: 999px;
		background: rgba(136, 192, 208, 0.14);
		color: var(--nord8);
		font-size: 11px;
		font-weight: 600;
		line-height: 1;
		letter-spacing: 0.08em;
		text-transform: uppercase;
	}

	@keyframes sweep {
		0%,
		100% {
			transform: scaleY(0.15);
			opacity: 0.5;
		}
		50% {
			transform: scaleY(0.9);
			opacity: 1;
		}
	}
</style>
