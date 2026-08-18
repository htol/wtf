<script lang="ts">
	import { invoke } from '@tauri-apps/api/core';
	import { listen } from '@tauri-apps/api/event';
	import { onMount } from 'svelte';

	interface Settings {
		language: string;
		gpu_device: number;
		use_gpu: boolean;
		model_path: string | null;
		model_id: string | null;
		silence_peak: number;
		overlay_x: number;
		overlay_y: number;
	}

	interface ModelInfo {
		id: string;
		file: string;
		installed: boolean;
		size_bytes: number | null;
		active: boolean;
	}

	interface GpuDevice {
		index: number;
		name: string;
		pci_bus_id: string;
	}

	interface DownloadProgress {
		id: string;
		downloaded: number;
		total: number | null;
		done: boolean;
	}

	const LANGUAGES: Array<{ value: string; label: string }> = [
		{ value: 'auto', label: 'Auto-detect' },
		{ value: 'en', label: 'English' },
		{ value: 'ru', label: 'Russian' }
	];

	let settings = $state<Settings | null>(null);
	let models = $state<ModelInfo[]>([]);
	let gpus = $state<GpuDevice[]>([]);
	let progress = $state<Record<string, DownloadProgress>>({});
	let savedAt = $state<string | null>(null);
	let rebindResult = $state<string | null>(null);

	let selectedModel = $derived(models.find((m) => m.id === settings?.model_id) ?? null);

	function formatSize(bytes: number | null): string {
		if (bytes === null) return '';
		if (bytes >= 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
		return `${Math.round(bytes / 1024 / 1024)} MB`;
	}

	async function refreshModels() {
		models = await invoke<ModelInfo[]>('list_models', {
			manualPath: settings?.model_path ?? null,
			modelId: settings?.model_id ?? null
		});
	}

	// Model and GPU pickers save immediately: a choice without a download
	// (or vice versa) would otherwise leave the active model ambiguous.
	// Each write re-fetches settings and copies only the fields this tab
	// owns: writing a stale full copy would wipe slices other tabs own
	// (prompts, overlay position).
	async function persistOwn() {
		if (!settings) return;
		const current = await invoke<Settings>('get_settings');
		current.language = settings.language;
		current.gpu_device = settings.gpu_device;
		current.use_gpu = settings.use_gpu;
		current.model_path = settings.model_path;
		current.model_id = settings.model_id;
		current.silence_peak = settings.silence_peak;
		await invoke('set_settings', { settings: current });
	}

	async function pickModel(id: string | null) {
		if (!settings) return;
		settings.model_id = id;
		await persistOwn();
		await refreshModels();
	}

	async function pickGpu(index: number) {
		if (!settings) return;
		settings.gpu_device = index;
		await persistOwn();
	}

	async function download() {
		if (!selectedModel) return;
		const id = selectedModel.id;
		progress[id] = { id, downloaded: 0, total: null, done: false };
		try {
			await invoke('download_model', { modelId: id });
		} catch (e) {
			alert(`Download failed: ${e}`);
		}
		delete progress[id];
		await refreshModels();
	}

	async function removeModel() {
		if (!selectedModel || !confirm(`Delete ${selectedModel.id} from disk?`)) return;
		await invoke('delete_model', { modelId: selectedModel.id });
		await refreshModels();
	}

	async function openModelsDir() {
		await invoke('open_models_dir');
	}

	async function rebind() {
		rebindResult = null;
		try {
			await invoke('rebind_shortcuts');
			rebindResult = 'bound';
		} catch (e) {
			rebindResult = `${e}`;
		}
	}

	async function save() {
		if (!settings) return;
		await persistOwn();
		savedAt = new Date().toLocaleTimeString();
		await refreshModels();
	}

	onMount(() => {
		invoke<Settings>('get_settings').then((s) => {
			settings = s;
			refreshModels();
		});
		invoke<GpuDevice[]>('list_gpu_devices').then((devices) => (gpus = devices));
		const unlisten = listen<DownloadProgress>('model-download', (event) => {
			progress[event.payload.id] = event.payload;
		});
		return () => {
			unlisten.then((u) => u());
		};
	});
</script>

{#if settings}
	<form onsubmit={(e) => { e.preventDefault(); save(); }}>
		<section>
			<h2>Dictation</h2>
			<label>
				Language
				<select bind:value={settings.language}>
					{#each LANGUAGES as lang (lang.value)}
						<option value={lang.value}>{lang.label}</option>
					{/each}
				</select>
			</label>
			<label>
				<input type="checkbox" bind:checked={settings.use_gpu} />
				Run inference on GPU
			</label>
			{#if gpus.length > 0}
				<label>
					GPU device
					<select value={settings.gpu_device} onchange={(e) => pickGpu(Number(e.currentTarget.value))}>
						{#each gpus as gpu (gpu.index)}
							<option value={gpu.index}>
								{gpu.name} — {gpu.pci_bus_id}
							</option>
						{/each}
					</select>
				</label>
			{:else}
				<label>
					GPU device index
					<input type="number" min="0" bind:value={settings.gpu_device} disabled={!settings.use_gpu} />
				</label>
			{/if}
			<label>
				Silence threshold (0–1; recordings below it are skipped, 0 = off)
				<input
					type="number"
					min="0"
					max="1"
					step="0.01"
					bind:value={settings.silence_peak}
				/>
			</label>
		</section>

		<section>
			<h2>Model</h2>
			<div class="row">
				<select value={settings.model_id ?? ''} onchange={(e) => pickModel(e.currentTarget.value || null)}>
					<option value="">Auto (latest downloaded)</option>
					{#each models as model (model.id)}
						<option value={model.id}>
							{model.id}{model.installed ? ` (${formatSize(model.size_bytes)})` : ''}
						</option>
					{/each}
				</select>
				{#if selectedModel}
					{#if progress[selectedModel.id] && !progress[selectedModel.id].done}
						{@const p = progress[selectedModel.id]}
						<div class="progress">
							<div class="bar" style="width: {p.total ? Math.min(100, (p.downloaded / p.total) * 100) : 0}%"></div>
							<span>{formatSize(p.downloaded)}{p.total ? ` / ${formatSize(p.total)}` : ''}</span>
						</div>
					{:else}
						{#if selectedModel.installed}
							<span class="installed" title="Downloaded">
								<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12" /></svg>
							</span>
						{/if}
						<button type="button" class="icon" title={selectedModel.installed ? 'Re-download' : 'Download'} onclick={() => download()}>
							{#if selectedModel.installed}
								<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10" /><polyline points="1 20 1 14 7 14" /><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" /></svg>
							{:else}
								<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="7 10 12 15 17 10" /><line x1="12" y1="15" x2="12" y2="3" /></svg>
							{/if}
						</button>
						{#if selectedModel.installed}
							<button type="button" class="icon danger" title="Delete from disk" onclick={() => removeModel()}>
								<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6" /><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" /><line x1="10" y1="11" x2="10" y2="17" /><line x1="14" y1="11" x2="14" y2="17" /></svg>
							</button>
						{/if}
						<button type="button" class="icon" title="Show models folder" onclick={() => openModelsDir()}>
							<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" /></svg>
						</button>
					{/if}
				{/if}
			</div>
			<label>
				Manual model path (overrides the picker)
				<input
					type="text"
					placeholder="/path/to/ggml-model.bin"
					value={settings.model_path ?? ''}
					onchange={(e) => (settings!.model_path = e.currentTarget.value || null)}
				/>
			</label>
		</section>

		<section>
		<h2>Shortcuts</h2>
		<p class="hint">
			Global shortcuts are bound via the desktop portal (KDE). Rebinding opens the
			Plasma dialog for both shortcuts.
		</p>
		<button type="button" onclick={() => rebind()}>Rebind shortcuts</button>
		{#if rebindResult}<span class="muted">{rebindResult}</span>{/if}
	</section>

	<footer class="actions">
			<button type="submit" class="primary">Save</button>
			{#if savedAt}<span class="muted">saved at {savedAt}</span>{/if}
		</footer>
	</form>
{:else}
	<p class="hint">Loading settings…</p>
{/if}

<style>
	form {
		display: flex;
		flex-direction: column;
		gap: 24px;
	}

	section {
		display: flex;
		flex-direction: column;
		gap: 10px;
	}

	h2 {
		margin: 0;
		font-size: 13px;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.08em;
		color: var(--nord8);
	}

	label {
		display: flex;
		align-items: center;
		gap: 10px;
		color: var(--nord4);
		max-width: 520px;
	}

	select,
	input[type='text'],
	input[type='number'] {
		font: inherit;
		color: var(--nord6);
		background: var(--nord1);
		border: 1px solid var(--nord3);
		border-radius: 4px;
		padding: 5px 8px;
	}

	select option {
		background: var(--nord0);
		color: var(--nord4);
	}

	input[type='number'] {
		width: 70px;
	}

	input[type='text'] {
		flex: 1;
	}

	.row {
		display: flex;
		align-items: center;
		gap: 10px;
		max-width: 560px;
	}

	.row > select {
		flex: 1;
	}

	.icon {
		padding: 4px 6px;
		display: inline-flex;
		align-items: center;
		border-color: transparent;
		background: transparent;
	}

	.icon:hover {
		background: var(--nord2);
	}

	.installed {
		color: var(--nord14);
		display: inline-flex;
		align-items: center;
	}

	.danger {
		color: var(--nord11);
	}

	.danger:hover {
		background: var(--nord11);
		color: var(--nord6);
	}

	.muted {
		color: var(--nord3);
	}

	.hint {
		color: var(--nord3);
		margin: 0;
	}

	.progress {
		display: flex;
		align-items: center;
		gap: 8px;
		min-width: 180px;
	}

	.bar {
		height: 6px;
		border-radius: 3px;
		background: var(--nord10);
		transition: width 0.2s;
		min-width: 2px;
	}

	.progress > span {
		font-size: 12px;
		color: var(--nord3);
		white-space: nowrap;
	}

	.actions {
		display: flex;
		align-items: center;
		gap: 12px;
	}
</style>
