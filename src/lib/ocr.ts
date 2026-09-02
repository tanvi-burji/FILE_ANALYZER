import { createWorker, type Worker } from 'tesseract.js'

let workerPromise: Promise<Worker> | null = null
let lastError: string | null = null
let workerBlobUrl: string | null = null

export function getLastOcrError(): string | null {
  return lastError
}

function formatErr(err: unknown): string {
  if (err == null) return 'Unknown OCR worker error'
  if (typeof err === 'string' && err.trim()) return err
  if (err instanceof Error) {
    return err.message || err.name || String(err)
  }
  if (typeof err === 'object') {
    const anyErr = err as { message?: string; type?: string; filename?: string }
    if (anyErr.message) return anyErr.message
    try {
      return JSON.stringify(err)
    } catch {
      return String(err)
    }
  }
  return String(err)
}

function absAsset(path: string): string {
  return `${window.location.origin}/ocr-assets/${path}`
}

/**
 * Load worker script into a Blob URL so we don't rely on importScripts(workerPath)
 * from a tiny wrapper blob (that pattern breaks under Vite/COEP).
 */
async function makeInlineWorkerUrl(): Promise<string> {
  if (workerBlobUrl) return workerBlobUrl
  const res = await fetch(absAsset('worker.min.js'))
  if (!res.ok) {
    throw new Error(`Could not load OCR worker script (${res.status})`)
  }
  const code = await res.text()
  if (!code || code.length < 100) {
    throw new Error('OCR worker script was empty or incomplete')
  }
  workerBlobUrl = URL.createObjectURL(
    new Blob([code], { type: 'application/javascript' }),
  )
  return workerBlobUrl
}

async function startWorker(options: {
  workerPath: string
  corePath: string
  langPath: string
  workerBlobURL: boolean
}): Promise<Worker> {
  try {
    return await createWorker('eng', 1, {
      ...options,
      gzip: true,
      logger: () => undefined,
      errorHandler: (err) => {
        lastError = formatErr(err)
      },
    })
  } catch (err) {
    throw new Error(formatErr(err))
  }
}

async function getWorker(): Promise<Worker> {
  if (!workerPromise) {
    workerPromise = (async () => {
      const attempts: Array<() => Promise<Worker>> = [
        // 1) Inline worker blob + local core/lang (best for Vite + COEP)
        async () => {
          const workerPath = await makeInlineWorkerUrl()
          return startWorker({
            workerPath,
            corePath: absAsset('tesseract-core-simd-lstm.wasm.js'),
            langPath: absAsset(''),
            workerBlobURL: false,
          })
        },
        // 2) Same with non-SIMD core
        async () => {
          const workerPath = await makeInlineWorkerUrl()
          return startWorker({
            workerPath,
            corePath: absAsset('tesseract-core-lstm.wasm.js'),
            langPath: absAsset(''),
            workerBlobURL: false,
          })
        },
        // 3) Default CDN paths (credentialless COEP allows this)
        async () =>
          startWorker({
            workerPath:
              'https://cdn.jsdelivr.net/npm/tesseract.js@5.1.1/dist/worker.min.js',
            corePath: 'https://cdn.jsdelivr.net/npm/tesseract.js-core@5.0.0',
            langPath:
              'https://cdn.jsdelivr.net/npm/@tesseract.js-data/eng/4.0.0_best_int',
            workerBlobURL: true,
          }),
      ]

      const errors: string[] = []
      for (const attempt of attempts) {
        try {
          return await attempt()
        } catch (err) {
          errors.push(formatErr(err))
        }
      }

      lastError = errors.filter(Boolean).join(' | ') || 'OCR worker failed to start'
      throw new Error(`Failed to start OCR engine: ${lastError}`)
    })().catch((err) => {
      workerPromise = null
      lastError = formatErr(err)
      throw err instanceof Error ? err : new Error(`Failed to start OCR engine: ${lastError}`)
    })
  }
  return workerPromise
}

async function toOcrInput(
  source: HTMLCanvasElement | Blob | File | string,
): Promise<Blob | File | string> {
  if (typeof HTMLCanvasElement !== 'undefined' && source instanceof HTMLCanvasElement) {
    return new Promise<Blob>((resolve, reject) => {
      source.toBlob(
        (blob) => {
          if (!blob) reject(new Error('Could not encode image for OCR.'))
          else resolve(blob)
        },
        'image/png',
        0.95,
      )
    })
  }
  return source as Blob | File | string
}

const MAX_OCR_SIDE = 1280

async function normalizeImageFile(file: Blob): Promise<Blob> {
  if (typeof createImageBitmap !== 'function') return file
  try {
    const bitmap = await createImageBitmap(file)
    const scale = Math.min(1, MAX_OCR_SIDE / Math.max(bitmap.width, bitmap.height))
    const canvas = document.createElement('canvas')
    canvas.width = Math.max(1, Math.round(bitmap.width * scale))
    canvas.height = Math.max(1, Math.round(bitmap.height * scale))
    const ctx = canvas.getContext('2d')
    if (!ctx) {
      bitmap.close()
      return file
    }
    ctx.fillStyle = '#ffffff'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height)
    bitmap.close()
    const blob = await new Promise<Blob | null>((resolve) =>
      canvas.toBlob(resolve, 'image/jpeg', 0.85),
    )
    canvas.width = 0
    canvas.height = 0
    return blob || file
  } catch {
    return file
  }
}

export async function ocrImageSource(
  source: HTMLCanvasElement | Blob | File | string,
): Promise<string> {
  lastError = null
  let input = await toOcrInput(source)
  if (input instanceof Blob) {
    input = await normalizeImageFile(input)
  }

  const worker = await getWorker()
  try {
    const result = await worker.recognize(input)
    return (result.data.text || '')
      .replace(/\r\n/g, '\n')
      .replace(/[ \t]+\n/g, '\n')
      .replace(/\n{3,}/g, '\n\n')
      .trim()
  } catch (err) {
    lastError = formatErr(err)
    try {
      await worker.terminate()
    } catch {
      // ignore
    }
    workerPromise = null
    throw new Error(lastError || 'OCR failed on this image.')
  }
}

export async function terminateOcrWorker(): Promise<void> {
  if (!workerPromise) return
  try {
    const worker = await workerPromise
    await worker.terminate()
  } catch {
    // ignore
  } finally {
    workerPromise = null
  }
  if (workerBlobUrl) {
    URL.revokeObjectURL(workerBlobUrl)
    workerBlobUrl = null
  }
}
