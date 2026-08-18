import { svelte } from '@sveltejs/vite-plugin-svelte';
import { defineConfig } from 'vite';
import { resolve } from 'node:path';

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
		sourcemap: !!process.env.TAURI_DEBUG,
		rollupOptions: {
			input: {
				main: resolve(__dirname, 'index.html'),
				overlay: resolve(__dirname, 'overlay.html')
			}
		}
	}
});
