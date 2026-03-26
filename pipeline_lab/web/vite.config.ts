import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { PIPELINE_LAB_LOCAL } from './src/siteUrls'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

export default defineConfig({
  plugins: [react()],
  /** Only crawl the SPA entry so `public/pose_3d_viewer.html` (CDN import map + `three`) is not scanned. */
  optimizeDeps: {
    entries: [path.resolve(__dirname, 'index.html')],
  },
  server: {
    host: 'localhost',
    /** Single dev UI URL: always http://localhost:5173 (fail if port busy). */
    port: 5173,
    strictPort: true,
    proxy: {
      '/api': PIPELINE_LAB_LOCAL.apiOrigin,
    },
  },
})
