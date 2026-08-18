<script lang="ts">
	import { invoke } from '@tauri-apps/api/core';
	import { onMount } from 'svelte';

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
	let savedAt = $state<string | null>(null);

	let selected = $derived(prompts.find((p) => p.name === selectedName) ?? null);

	async function load() {
		const settings = await invoke<Settings>('get_settings');
		prompts = settings.prompts;
		active = settings.active_prompt;
		selectedName = settings.prompts[0]?.name ?? null;
	}

	async function persist() {
		// The Prompts tab owns its slice of settings; the rest is untouched.
		const settings = await invoke<Settings>('get_settings');
		settings.prompts = prompts;
		settings.active_prompt = active;
		await invoke('set_settings', { settings });
		savedAt = new Date().toLocaleTimeString();
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
		load();
	});
</script>

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
			<div class="row save-row">
				<button type="button" class="primary" onclick={() => persist()}>Save</button>
				{#if savedAt}<span class="muted">saved at {savedAt}</span>{/if}
			</div>
		</div>
	{:else}
		<p class="hint">No prompt selected. Create one, write a few example sentences, activate it.</p>
	{/if}
</div>

<style>
	.prompts {
		display: flex;
		gap: 20px;
		height: 100%;
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

	.save-row {
		justify-content: flex-start;
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

	.muted {
		color: var(--nord3);
	}

	.hint {
		color: var(--nord3);
	}
</style>
