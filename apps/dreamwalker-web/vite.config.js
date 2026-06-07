import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  base: process.env.DREAMWALKER_BASE || '/',
  plugins: [react()],
  build: {
    outDir: process.env.DREAMWALKER_OUT_DIR || 'dist',
    emptyOutDir: true,
    rollupOptions: {
      input: {
        main: fileURLToPath(new URL('./index.html', import.meta.url)),
        overlay: fileURLToPath(new URL('./overlay.html', import.meta.url))
      },
      output: {
        manualChunks(id) {
          if (id.includes('sync-ammo')) {
            return 'physics';
          }

          if (id.includes('@playcanvas/react') || id.includes('/playcanvas/')) {
            return 'playcanvas';
          }

          if (id.includes('/react/') || id.includes('react-dom')) {
            return 'react-vendor';
          }

          return null;
        }
      }
    }
  },
  server: {
    host: true,
    port: 5173,
    allowedHosts: ['neighbor-encyclopedia-modern-component.trycloudflare.com']
  }
});
