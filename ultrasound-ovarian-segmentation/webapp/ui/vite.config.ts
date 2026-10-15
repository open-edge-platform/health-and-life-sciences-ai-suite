import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The production build is emitted to webapp/ui/dist and served by the Flask
// app (same origin, so /api and the SSE stream work directly). In dev, the
// Vite server proxies /api to the Flask control plane on port 8080.
export default defineConfig({
  plugins: [react()],
  base: '/',
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    strictPort: false,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8080',
        changeOrigin: true,
      },
    },
  },
});
