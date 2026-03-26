/**
 * Canonical local URLs for Pipeline Lab. Keep in sync with `vite.config.ts` (proxy target + port).
 *
 * - **Backend (API):** always `apiOrigin` — uvicorn listens here.
 * - **Frontend (browser):** either `bundledOrigin` (built UI served by uvicorn, same tab as API) or
 *   `devUiOrigin` (Vite HMR; `/api` is proxied to `apiOrigin`, so you only open one tab).
 */
export const PIPELINE_LAB_LOCAL = {
  apiOrigin: 'http://localhost:8765',
  /** Vite dev server (fixed port 5173 in vite.config.ts). */
  devUiOrigin: 'http://localhost:5173',
  /** Built SPA + API when `PIPELINE_LAB_WEB_DIST` is set — same URL as the API. */
  bundledOrigin: 'http://localhost:8765',
} as const
