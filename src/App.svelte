<script lang="ts">
	import Settings from './lib/Settings.svelte';
	import History from './lib/History.svelte';

	type Tab = 'settings' | 'history';
	let tab: Tab = $state('settings');
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
</style>
