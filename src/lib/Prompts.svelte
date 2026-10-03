<script lang="ts">
	import { invoke } from '@tauri-apps/api/core';
	import { onMount } from 'svelte';
	import { releaseSaveBar, saveBar } from './saveState';

	interface NamedPrompt {
		name: string;
		text: string;
	}

	interface Settings {
		prompts: NamedPrompt[];
		active_prompt: string | null;
		[key: string]: unknown;
	}

	let prompts = $state<NamedPrompt[]>([]);
	let active = $state<string | null>(null);
	let selectedName = $state<string | null>(null);
	// Snapshot of the persisted slice (prompts + active) for dirty
	// tracking; divergence enables the tab-bar Save button. `loaded` keeps
	// the button inert until the initial get_settings resolves.
	let snapshot = $state('');
	let loaded = $state(false);

	let selected = $derived(prompts.find((p) => p.name === selectedName) ?? null);

	function ownedValues(): string {
		return JSON.stringify([prompts, active]);
	}

	const dirty = $derived(loaded && ownedValues() !== snapshot);

	$effect(() => {
		saveBar.update((s) => ({ ...s, dirty }));
	});

	async function load() {
		const settings = await invoke<Settings>('get_settings');
		prompts = settings.prompts;
		active = settings.active_prompt;
		selectedName = settings.prompts[0]?.name ?? null;
		snapshot = ownedValues();
		loaded = true;
	}

	async function save() {
		if (!dirty) return;
		// The Prompts tab owns its slice of settings; the rest is untouched.
		const settings = await invoke<Settings>('get_settings');
		settings.prompts = prompts;
		settings.active_prompt = active;
		await invoke('set_settings', { settings });
		snapshot = ownedValues();
		saveBar.update((s) => ({ ...s, savedAt: new Date().toLocaleTimeString() }));
	}

	function select(name: string) {
		selectedName = name;
	}

	async function add() {
		const name = `prompt ${prompts.length + 1}`;
		prompts = [...prompts, { name, text: '' }];
		selectedName = name;
	}

	async function remove(name: string) {
		prompts = prompts.filter((p) => p.name !== name);
		if (active === name) active = null;
		if (selectedName === name) selectedName = prompts[0]?.name ?? null;
	}

	async function activate(name: string) {
		active = active === name ? null : name;
	}

	// Populate the list from settings on mount: without this the tab always
	// starts empty and saved prompts look lost (the actual bug behind
	// "prompts disappear after restart").
	onMount(() => {
		const owner = () => void save();
		saveBar.update((s) => ({ ...s, save: owner }));
		load();
		return () => releaseSaveBar(owner);
	});
</script>

<div class="wrap">
	<p class="banner">
		Prompts apply to the Whisper engine only; GigaAM and Qwen3-ASR do not
		support prompts.
	</p>
	<div class="prompts">
		<div class="list">
			<button type="button" class="primary add" onclick={() => add()}>+ New prompt</button>
			{#each prompts as prompt (prompt.name)}
				<div
					class="item"
					class:selected={prompt.name === selectedName}
					onclick={() => select(prompt.name)}
					onkeydown={(e) => e.key === 'Enter' && select(prompt.name)}
					role="button"
					tabindex="0"
				>
					<span class="name">{prompt.name}</span>
					{#if active === prompt.name}
						<span class="badge">active</span>
					{/if}
				</div>
			{/each}
		</div>

		{#if selected}
			<div class="editor">
				<div class="row">
					<input
						type="text"
						bind:value={selected.name}
						onchange={() => {
							if (active === selectedName) active = selected.name;
							selectedName = selected.name;
						}}
					/>
					<button
						type="button"
						class={active === selected.name ? '' : 'primary'}
						onclick={() => activate(selected.name)}
					>
						{active === selected.name ? 'Deactivate' : 'Activate'}
					</button>
					<button type="button" class="danger" onclick={() => remove(selected.name)}>Delete</button>
				</div>
				<textarea
					bind:value={selected.text}
					placeholder="Example sentences in the style you want transcribed, e.g. mixed Russian/English speech. This conditions the decoder's style and vocabulary — it is not an instruction the model follows."
				></textarea>
			</div>
		{:else}
			<p class="hint">No prompt selected. Create one, write a few example sentences, activate it.</p>
		{/if}
	</div>
</div>

<style>
	.wrap {
		display: flex;
		flex-direction: column;
		gap: 12px;
		height: 100%;
	}

	.banner {
		margin: 0;
		padding: 8px 12px;
		border: 1px solid var(--nord2);
		border-radius: 6px;
		color: var(--nord9);
		font-size: 12px;
	}

	.prompts {
		display: flex;
		gap: 20px;
		height: 100%;
		flex: 1;
		min-height: 0;
	}

	.list {
		display: flex;
		flex-direction: column;
		gap: 6px;
		width: 180px;
		flex-shrink: 0;
	}

	.add {
		margin-bottom: 6px;
	}

	.item {
		display: flex;
		align-items: center;
		gap: 8px;
		padding: 6px 10px;
		border-radius: 4px;
		border: 1px solid transparent;
		cursor: pointer;
		color: var(--nord4);
	}

	.item:hover {
		background: var(--nord1);
	}

	.item.selected {
		background: var(--nord1);
		border-color: var(--nord3);
	}

	.name {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.badge {
		font-size: 10px;
		text-transform: uppercase;
		letter-spacing: 0.06em;
		color: var(--nord7);
		margin-left: auto;
	}

	.editor {
		display: flex;
		flex-direction: column;
		gap: 10px;
		flex: 1;
		max-width: 520px;
	}

	.row {
		display: flex;
		gap: 8px;
		align-items: center;
	}

	.row input {
		flex: 1;
	}

	input[type='text'],
	textarea {
		font: inherit;
		color: var(--nord4);
		background: var(--nord1);
		border: 1px solid var(--nord3);
		border-radius: 4px;
		padding: 6px 8px;
	}

	textarea {
		flex: 1;
		resize: none;
		min-height: 160px;
	}

	.danger {
		color: var(--nord11);
		border-color: var(--nord11);
	}

	.danger:hover {
		background: var(--nord11);
		color: var(--nord6);
	}

	.hint {
		color: var(--nord3);
	}
</style>
