import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { scanaskApiPlugin } from './server/apiPlugin.ts'
import { tesseractAssetsPlugin } from './server/tesseractAssetsPlugin.ts'

export default defineConfig({
  plugins: [react(), scanaskApiPlugin(), tesseractAssetsPlugin()],
  server: {
    headers: {
      'Cache-Control': 'no-store',
      'Cross-Origin-Opener-Policy': 'same-origin',
      // credentialless keeps SharedArrayBuffer for WebLLM while allowing
      // Tesseract language/core assets to load cross-origin for OCR
      'Cross-Origin-Embedder-Policy': 'credentialless',
    },
  },
  preview: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'credentialless',
    },
  },
  optimizeDeps: {
    exclude: ['@mlc-ai/web-llm'],
    include: ['tesseract.js'],
  },
  assetsInclude: ['**/*.wasm'],
})
