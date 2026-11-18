import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

// `npm run dev` forwards API calls to a running server. API_TARGET can point it elsewhere.
const apiTarget = process.env.API_TARGET || 'http://127.0.0.1:8000';
const forwarded = { target: apiTarget, changeOrigin: true };

export default defineConfig(({ command }) => ({
  plugins: [react(), tailwindcss()],
  // FastAPI serves the build from /static.
  base: command === 'build' ? '/static/' : '/',
  build: {
    outDir: '../src/speech_grammar/api/static',
    emptyOutDir: true,
    assetsInlineLimit: 0, // the server's CSP refuses data: fonts
  },
  server: {
    proxy: { '/v1': forwarded, '/docs': forwarded, '/openapi.json': forwarded, '/readyz': forwarded },
  },
}));
