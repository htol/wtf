<script lang="ts">
	import { invoke } from '@tauri-apps/api/core';
	import { listen } from '@tauri-apps/api/event';
	import { onMount } from 'svelte';

	interface Entry {
		id: number;
		text: string;
		language: string;
		created_at: number; // unix seconds
	}

	let entries = $state<Entry[]>([]);
	let selected = $state<Set<number>>(new Set());
	let copiedId = $state<number | null>(null);

	let allSelected = $derived(entries.length > 0 && selected.size === entries.length);

	function formatTime(unixSeconds: number): string {
		return new Date(unixSeconds * 1000).toLocaleString();
	}

	async function refresh() {
		entries = await invoke<Entry[]>('list_history', { limit: 100 });
		selected = new Set();
	}

	async function copy(entry: Entry) {
		await invoke('copy_to_clipboard', { text: entry.text });
		copiedId = entry.id;
		setTimeout(() => (copiedId = null), 1200);
	}

	function toggle(id: number) {
		const next = new Set(selected);
		if (next.has(id)) {
			next.delete(id);
		} else {
			next.add(id);
		}
		selected = next;
	}

	function toggleAll() {
		selected = allSelected ? new Set() : new Set(entries.map((e) => e.id));
	}

	async function remove(ids: number[]) {
		if (ids.length === 0) return;
		await invoke('delete_history', { ids });
		await refresh();
	}

	onMount(() => {
		refresh();
		const unlisten = listen('transcript', () => {
			refresh();
		});
		return () => {
			unlisten.then((u) => u());
		};
	});
</script>

{#if entries.length === 0}
	<p class="hint">No transcriptions yet — press the record hotkey, speak, press again.</p>
{:else}
	<div class="toolbar">
		<label class="select-all">
			<input type="checkbox" checked={allSelected} onchange={toggleAll} />
			Select all
		</label>
		<button
			type="button"
			class="danger"
			onclick={() => remove([...selected])}
			disabled={selected.size === 0}
		>
			Delete selected{selected.size > 0 ? ` (${selected.size})` : ''}
		</button>
	</div>
	<ul class="entries">
		{#each entries as entry (entry.id)}
			<li class:selected={selected.has(entry.id)}>
				<div class="meta">
					<input
						type="checkbox"
						checked={selected.has(entry.id)}
						onchange={() => toggle(entry.id)}
					/>
					<span class="lang">{entry.language}</span>
					<span class="time">{formatTime(entry.created_at)}</span>
					<button type="button" class="icon danger" title="Delete" onclick={() => remove([entry.id])}>
						<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
							<polyline points="3 6 5 6 21 6" />
							<path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
							<line x1="10" y1="11" x2="10" y2="17" />
							<line x1="14" y1="11" x2="14" y2="17" />
						</svg>
					</button>
					<button type="button" class="copy" onclick={() => copy(entry)}>
						{copiedId === entry.id ? 'Copied' : 'Copy'}
					</button>
				</div>
				<p class="text">{entry.text}</p>
			</li>
		{/each}
	</ul>
{/if}

<style>
	.toolbar {
		display: flex;
		align-items: center;
		gap: 10px;
		margin-bottom: 14px;
		max-width: 640px;
	}

	.select-all {
		display: flex;
		align-items: center;
		gap: 6px;
		color: var(--nord3);
		margin-right: auto;
	}

	.entries {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-direction: column;
		gap: 14px;
		max-width: 640px;
	}

	li.selected .text {
		border-left-color: var(--nord10);
	}

	.meta {
		display: flex;
		align-items: center;
		gap: 10px;
	}

	.lang {
		font-size: 11px;
		text-transform: uppercase;
		letter-spacing: 0.06em;
		color: var(--nord7);
	}

	.time {
		font-size: 12px;
		color: var(--nord3);
	}

	.copy {
		font-size: 12px;
		padding: 2px 10px;
		margin-left: 8px;
	}

	.icon {
		padding: 3px 5px;
		display: inline-flex;
		align-items: center;
	}

	.danger {
		color: var(--nord11);
		border-color: var(--nord11);
	}

	.danger:hover {
		background: var(--nord11);
		color: var(--nord6);
	}

	button.danger:disabled {
		color: var(--nord3);
		border-color: var(--nord3);
		cursor: default;
	}

	button.danger:disabled:hover {
		background: var(--nord1);
		color: var(--nord3);
	}

	.text {
		margin: 4px 0 0;
		color: var(--nord5);
		white-space: pre-wrap;
		border-left: 2px solid var(--nord1);
		padding-left: 10px;
	}

	.hint {
		color: var(--nord3);
	}
</style>
