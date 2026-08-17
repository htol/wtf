import { svelte } from '@sveltejs/vite-plugin-svelte';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [svelte()],
	clearScreen: false,
	server: {
		port: 5173,
		strictPort: true
	},
	envPrefix: 'TAURI_',
	build: {
		target: 'chrome105',
		sourcemap: !!process.env.TAURI_DEBUG
	}
});
