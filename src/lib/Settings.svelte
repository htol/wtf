<script lang="ts">
	import { invoke } from '@tauri-apps/api/core';
	import { listen } from '@tauri-apps/api/event';
	import { onMount } from 'svelte';
	import { releaseSaveBar, saveBar } from './saveState';

	interface Settings {
		engine: Engine;
		language: string;
		gpu_device: number;
		use_gpu: boolean;
		model_path: string | null;
		model_id: string | null;
		gigaam_model_id: string | null;
		qwen_model_id: string | null;
		silence_peak: number;
		overlay_x: number;
		overlay_y: number;
		start_hidden: boolean;
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

	// A catalog card of the GigaAM or Qwen3-ASR engine.
	interface CardModelInfo {
		id: string;
		file: string;
		note: string;
		installed: boolean;
		size_bytes: number | null;
		approx_bytes: number;
		active: boolean;
	}

	interface DownloadProgress {
		id: string;
		downloaded: number;
		total: number | null;
		done: boolean;
	}

	interface UpdateStatus {
		id: string;
		file: string;
		up_to_date: boolean;
	}

	type Engine = 'whisper' | 'gigaam' | 'qwen';
	// Engines with a card catalog; the value doubles as the infix of their
	// tauri commands and settings field.
	type CardEngine = 'gigaam' | 'qwen';

	const ENGINES: Array<{ value: Engine; label: string }> = [
		{ value: 'whisper', label: 'Whisper' },
		{ value: 'gigaam', label: 'GigaAM' },
		{ value: 'qwen', label: 'Qwen3-ASR' }
	];

	const LANGUAGES: Array<{ value: string; label: string }> = [
		{ value: 'auto', label: 'Auto' },
		{ value: 'en', label: 'EN' },
		{ value: 'ru', label: 'RU' }
	];

	type SubTab = 'general' | 'whisper' | 'gigaam' | 'qwen' | 'shortcuts';

	let subTab = $state<SubTab>('general');
	let settings = $state<Settings | null>(null);
	let models = $state<ModelInfo[]>([]);
	let cardModels = $state<Record<CardEngine, CardModelInfo[]>>({ gigaam: [], qwen: [] });
	let gpus = $state<GpuDevice[]>([]);
	let progress = $state<Record<string, DownloadProgress>>({});
	// Snapshot of the persisted values this tab owns; divergence from the
	// live form state is what enables the tab-bar Save button.
	let snapshot = $state('');
	let unloadedAt = $state<string | null>(null);
	let rebindResult = $state<string | null>(null);
	// Verdicts of the last "Check for updates" run: catalog id -> result.
	// Cleared per model after a successful re-download.
	let updates = $state<Record<string, boolean | undefined>>({});
	let checking = $state(false);

	let selectedModel = $derived(models.find((m) => m.id === settings?.model_id) ?? null);

	function ownedValues(): string {
		if (!settings) return '';
		return JSON.stringify([
			settings.engine,
			settings.language,
			settings.gpu_device,
			settings.use_gpu,
			settings.model_path,
			settings.model_id,
			settings.gigaam_model_id,
			settings.qwen_model_id,
			settings.silence_peak,
			settings.start_hidden
		]);
	}

	const dirty = $derived(settings !== null && ownedValues() !== snapshot);

	$effect(() => {
		saveBar.update((s) => ({ ...s, dirty }));
	});

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
		cardModels.gigaam = await invoke<CardModelInfo[]>('list_gigaam_models', {
			gigaamModelId: settings?.gigaam_model_id ?? null
		});
		cardModels.qwen = await invoke<CardModelInfo[]>('list_qwen_models', {
			qwenModelId: settings?.qwen_model_id ?? null
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
		current.engine = settings.engine;
		current.language = settings.language;
		current.gpu_device = settings.gpu_device;
		current.use_gpu = settings.use_gpu;
		current.model_path = settings.model_path;
		current.model_id = settings.model_id;
		current.gigaam_model_id = settings.gigaam_model_id;
		current.qwen_model_id = settings.qwen_model_id;
		current.silence_peak = settings.silence_peak;
		current.start_hidden = settings.start_hidden;
		await invoke('set_settings', { settings: current });
		// Everything just written matches disk again (model/GPU pickers
		// persist through this same path).
		snapshot = ownedValues();
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
			delete updates[id];
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

	function cardProgress(id: string) {
		return progress[id] && !progress[id].done ? progress[id] : null;
	}

	async function downloadCard(engine: CardEngine, id: string) {
		progress[id] = { id, downloaded: 0, total: null, done: false };
		try {
			await invoke(`download_${engine}_model`, { modelId: id });
			delete updates[id];
		} catch (e) {
			alert(`Download failed: ${e}`);
		}
		delete progress[id];
		await refreshModels();
	}

	async function removeCard(engine: CardEngine, id: string) {
		const model = cardModels[engine].find((m) => m.id === id);
		if (!model || !confirm(`Delete ${model.id} from disk?`)) return;
		await invoke(`delete_${engine}_model`, { modelId: id });
		// The deleted card may have been the pinned pick.
		if (settings?.[`${engine}_model_id`] === id) {
			settings[`${engine}_model_id`] = null;
			await persistOwn();
		}
		await refreshModels();
	}

	async function pickCard(engine: CardEngine, id: string | null) {
		if (!settings) return;
		settings[`${engine}_model_id`] = id;
		await persistOwn();
		await refreshModels();
	}

	async function openModelsDir() {
		await invoke('open_models_dir');
	}

	// One run covers all engines (the catalogs are fetched and hashed
	// together). The single button lives on the General sub-tab; verdicts
	// surface in its report row and as badges on GigaAM/Qwen3-ASR cards.
	async function checkUpdates() {
		if (checking) return;
		checking = true;
		try {
			const statuses = await invoke<UpdateStatus[]>('check_model_updates');
			updates = Object.fromEntries(statuses.map((s) => [s.id, s.up_to_date]));
		} catch (e) {
			alert(`Update check failed: ${e}`);
		}
		checking = false;
	}

	async function unloadModel() {
		await invoke('unload_model');
		unloadedAt = new Date().toLocaleTimeString();
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
		if (!settings || !dirty) return;
		await persistOwn();
		const savedAt = new Date().toLocaleTimeString();
		saveBar.update((s) => ({ ...s, savedAt }));
		await refreshModels();
	}

	onMount(() => {
		const owner = () => void save();
		saveBar.update((s) => ({ ...s, save: owner }));
		invoke<Settings>('get_settings').then((s) => {
			settings = s;
			snapshot = ownedValues();
			refreshModels();
		});
		invoke<GpuDevice[]>('list_gpu_devices').then((devices) => (gpus = devices));
		const unlisten = listen<DownloadProgress>('model-download', (event) => {
			progress[event.payload.id] = event.payload;
		});
		return () => {
			releaseSaveBar(owner);
			unlisten.then((u) => u());
		};
	});
</script>

{#snippet modelCards(engine: CardEngine)}
	<div class="cards">
		<div
			class="card"
			class:active={settings?.[`${engine}_model_id`] === null && cardModels[engine].some((m) => m.installed)}
			role="button"
			tabindex="0"
			onclick={() => pickCard(engine, null)}
			onkeydown={(e) => e.key === 'Enter' && pickCard(engine, null)}
		>
			<span class="name">Auto</span>
			<span class="desc">the most recent download is used</span>
		</div>
		{#each cardModels[engine] as model (model.id)}
			{@const p = cardProgress(model.id)}
			<div
				class="card"
				class:active={model.active}
				role="button"
				tabindex="0"
				onclick={() => pickCard(engine, model.id)}
				onkeydown={(e) => e.key === 'Enter' && pickCard(engine, model.id)}
			>
				<div class="head">
					<span class="name">{model.id}</span>
					{#if updates[model.id] !== undefined}
						<span class="badge {updates[model.id] ? 'ok' : 'stale'}">
							{updates[model.id] ? 'up to date' : 'update available'}
						</span>
					{/if}
					{#if model.active}<span class="badge">active</span>{/if}
				</div>
				<span class="desc">{model.note}</span>
				<span class="size">
					{model.installed
						? formatSize(model.size_bytes)
						: `download ≈ ${formatSize(model.approx_bytes)}`}
				</span>
				{#if p}
					<div class="progress">
						<div
							class="bar"
							style="width: {p.total ? Math.min(100, (p.downloaded / p.total) * 100) : 0}%"
						></div>
						<span>{formatSize(p.downloaded)}{p.total ? ` / ${formatSize(p.total)}` : ''}</span>
					</div>
				{:else if model.installed}
					<div class="actions">
						<button
							type="button"
							class="icon"
							title="Re-download"
							onclick={() => downloadCard(engine, model.id)}
						>
							<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10" /><polyline points="1 20 1 14 7 14" /><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" /></svg>
						</button>
						<button
							type="button"
							class="icon danger"
							title="Delete from disk"
							onclick={() => removeCard(engine, model.id)}
						>
							<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6" /><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" /><line x1="10" y1="11" x2="10" y2="17" /><line x1="14" y1="11" x2="14" y2="17" /></svg>
						</button>
					</div>
				{:else}
					<div class="actions">
						<button
							type="button"
							class="icon"
							title="Download"
							onclick={() => downloadCard(engine, model.id)}
						>
							<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="7 10 12 15 17 10" /><line x1="12" y1="15" x2="12" y2="3" /></svg>
						</button>
					</div>
				{/if}
			</div>
		{/each}
	</div>
{/snippet}

{#if settings}
	<form onsubmit={(e) => { e.preventDefault(); save(); }}>
		<nav class="subnav">
			<button type="button" class:active={subTab === 'general'} onclick={() => (subTab = 'general')}>
				General
			</button>
			<button type="button" class:active={subTab === 'whisper'} onclick={() => (subTab = 'whisper')}>
				Whisper
			</button>
			<button type="button" class:active={subTab === 'gigaam'} onclick={() => (subTab = 'gigaam')}>
				GigaAM
			</button>
			<button type="button" class:active={subTab === 'qwen'} onclick={() => (subTab = 'qwen')}>
				Qwen3-ASR
			</button>
			<button
				type="button"
				class:active={subTab === 'shortcuts'}
				onclick={() => (subTab = 'shortcuts')}
			>
				Shortcuts
			</button>
		</nav>
		<div class="pane">
			{#if subTab === 'general'}
				<section>
					<h2>Dictation</h2>
					<label>
						Engine
						<select bind:value={settings.engine}>
							{#each ENGINES as engine (engine.value)}
								<option value={engine.value}>{engine.label}</option>
							{/each}
						</select>
					</label>
					<label>
						Language
						{#if settings.engine === 'gigaam'}
							<select disabled><option>RU</option></select>
						{:else}
							<select bind:value={settings.language}>
								{#each LANGUAGES as lang (lang.value)}
									<option value={lang.value}>{lang.label}</option>
								{/each}
							</select>
						{/if}
					</label>
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
					<h2>Startup</h2>
					<label>
						<input type="checkbox" bind:checked={settings.start_hidden} />
						Start with the window hidden in the tray
					</label>
					<p class="hint">
						The daemon keeps running either way; reopen the window from the
						tray menu ("Open wtf") or by launching the app again.
					</p>
				</section>

				<section>
					<h2>Model updates</h2>
					<div class="report">
						<button type="button" disabled={checking} onclick={() => checkUpdates()}>
							{checking ? 'Checking…' : 'Check for updates'}
						</button>
						{#each [...models, ...cardModels.gigaam, ...cardModels.qwen].filter((m) => m.installed && updates[m.id] !== undefined) as m (m.id)}
							<span class={updates[m.id] ? 'ok' : 'stale'}>
								{m.id}: {updates[m.id] ? 'up to date' : 'update available'}
							</span>
						{/each}
					</div>
					<p class="hint">
						Hashes installed models of all engines and compares them with huggingface.co.
					</p>
				</section>

				<section>
					<h2>Memory</h2>
					<div class="row">
						<button type="button" onclick={() => unloadModel()}>Unload models from memory</button>
						{#if unloadedAt}<span class="muted">unloaded at {unloadedAt}</span>{/if}
					</div>
					<p class="hint">
						Frees the weights of all engines from GPU/system memory; they reload on the
						next dictation. Switching models unloads automatically.
					</p>
				</section>
			{:else if subTab === 'whisper'}
				<section>
					<h2>Inference</h2>
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
										{gpu.pci_bus_id ? `${gpu.name} — ${gpu.pci_bus_id}` : gpu.name}
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
			{:else if subTab === 'gigaam'}
				<section>
					<p class="hint">
						Russian only, CPU-only; the language setting is ignored. Without a
						downloaded model the GigaAM engine falls back to Whisper.
					</p>
					{@render modelCards('gigaam')}
				</section>
			{:else if subTab === 'qwen'}
				<section>
					<p class="hint">
						Multilingual with language detection; runs on the GPU via llama.cpp
						(Vulkan), downloaded with the first model. Prompts are not supported.
					</p>
					{@render modelCards('qwen')}
				</section>
			{:else}
				<section>
					<p class="hint">
						Global shortcuts are bound via the desktop portal (KDE). Rebinding opens
						the Plasma dialog for both shortcuts.
					</p>
					<div class="row">
						<button type="button" onclick={() => rebind()}>Rebind shortcuts</button>
						{#if rebindResult}<span class="muted">{rebindResult}</span>{/if}
					</div>
				</section>
			{/if}
		</div>
	</form>
{:else}
	<p class="hint">Loading settings…</p>
{/if}

<style>
	form {
		display: flex;
		align-items: flex-start;
		gap: 24px;
	}

	/* Vertical sub-tab strip on the left edge of the Settings tab;
	 * stays in view while the pane next to it scrolls. */
	.subnav {
		display: flex;
		flex-direction: column;
		gap: 2px;
		width: 110px;
		flex-shrink: 0;
		position: sticky;
		top: 0;
	}

	.subnav button {
		text-align: left;
		border: none;
		background: transparent;
		color: var(--nord4);
		font-size: 12px;
		padding: 5px 10px;
		border-radius: 4px;
	}

	.subnav button:hover {
		background: var(--nord1);
	}

	.subnav button.active {
		background: var(--nord1);
		color: var(--nord8);
		font-weight: 600;
	}

	.pane {
		display: flex;
		flex-direction: column;
		gap: 24px;
		flex: 1;
		min-width: 0;
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

	.report {
		display: flex;
		align-items: center;
		flex-wrap: wrap;
		gap: 8px;
		max-width: 560px;
	}

	.report .ok {
		color: var(--nord14);
	}

	.report .stale {
		color: var(--nord13);
	}

	.card .badge.ok {
		color: var(--nord14);
		margin-left: 0;
	}

	.card .badge.stale {
		color: var(--nord13);
		margin-left: 0;
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

	.cards {
		display: flex;
		gap: 10px;
		flex-wrap: wrap;
	}

	.card {
		display: flex;
		flex-direction: column;
		gap: 6px;
		min-width: 200px;
		padding: 10px 12px;
		border: 1px solid var(--nord1);
		border-radius: 6px;
		cursor: pointer;
		color: var(--nord4);
	}

	.card:hover {
		background: var(--nord1);
	}

	.card.active {
		border-color: var(--nord10);
		background: var(--nord1);
	}

	.card .head {
		display: flex;
		align-items: center;
		gap: 8px;
	}

	.card .name {
		font-weight: 600;
		color: var(--nord6);
	}

	.card .badge {
		font-size: 10px;
		text-transform: uppercase;
		letter-spacing: 0.06em;
		color: var(--nord7);
		margin-left: auto;
	}

	.card .desc,
	.card .size {
		font-size: 12px;
		color: var(--nord3);
	}

	.card .actions {
		gap: 4px;
	}

	.card .progress {
		display: flex;
		align-items: center;
		gap: 8px;
		min-width: 160px;
	}
</style>
