import fs from 'node:fs'
import path from 'node:path'
import type { Connect, Plugin } from 'vite'

/**
 * Serve Tesseract worker/core/lang at /ocr-assets/* (same-origin).
 * Workers on a COEP page need matching COEP on the worker script response.
 */
export function tesseractAssetsPlugin(): Plugin {
  const root = process.cwd()

  const map: Record<string, string> = {
    '/ocr-assets/worker.min.js': path.join(
      root,
      'node_modules/tesseract.js/dist/worker.min.js',
    ),
    '/ocr-assets/tesseract-core-simd-lstm.wasm.js': path.join(
      root,
      'node_modules/tesseract.js-core/tesseract-core-simd-lstm.wasm.js',
    ),
    '/ocr-assets/tesseract-core-lstm.wasm.js': path.join(
      root,
      'node_modules/tesseract.js-core/tesseract-core-lstm.wasm.js',
    ),
    '/ocr-assets/tesseract-core-simd.wasm.js': path.join(
      root,
      'node_modules/tesseract.js-core/tesseract-core-simd.wasm.js',
    ),
    '/ocr-assets/tesseract-core.wasm.js': path.join(
      root,
      'node_modules/tesseract.js-core/tesseract-core.wasm.js',
    ),
  }

  const contentType = (file: string) => {
    if (file.endsWith('.js')) return 'application/javascript; charset=utf-8'
    if (file.endsWith('.wasm')) return 'application/wasm'
    if (file.endsWith('.gz')) return 'application/gzip'
    return 'application/octet-stream'
  }

  const setOcrHeaders = (res: Connect.ServerResponse, type: string) => {
    res.setHeader('Content-Type', type)
    res.setHeader('Cache-Control', 'public, max-age=31536000, immutable')
    res.setHeader('Cross-Origin-Resource-Policy', 'cross-origin')
    // Required so dedicated workers can run under the page's COEP
    res.setHeader('Cross-Origin-Embedder-Policy', 'credentialless')
  }

  const middleware: Connect.NextHandleFunction = (req, res, next) => {
    const url = req.url?.split('?')[0] || ''
    if (!url.startsWith('/ocr-assets/')) return next()

    if (url === '/ocr-assets/eng.traineddata.gz') {
      void (async () => {
        try {
          const upstream = await fetch(
            'https://cdn.jsdelivr.net/npm/@tesseract.js-data/eng/4.0.0_best_int/eng.traineddata.gz',
          )
          if (!upstream.ok) {
            res.statusCode = upstream.status
            res.end(`Failed to fetch lang data (${upstream.status})`)
            return
          }
          const buf = Buffer.from(await upstream.arrayBuffer())
          res.statusCode = 200
          setOcrHeaders(res, 'application/gzip')
          res.end(buf)
        } catch (err) {
          res.statusCode = 502
          res.end(err instanceof Error ? err.message : 'Lang data proxy failed')
        }
      })()
      return
    }

    const file = map[url]
    if (!file || !fs.existsSync(file)) {
      res.statusCode = 404
      res.end(`OCR asset not found: ${url}`)
      return
    }

    res.statusCode = 200
    setOcrHeaders(res, contentType(file))
    fs.createReadStream(file).pipe(res)
  }

  return {
    name: 'scanask-tesseract-assets',
    configureServer(server) {
      server.middlewares.use(middleware)
    },
    configurePreviewServer(server) {
      server.middlewares.use(middleware)
    },
  }
}
