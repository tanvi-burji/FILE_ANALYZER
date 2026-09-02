import * as pdfjs from 'pdfjs-dist'
import mammoth from 'mammoth'
import type { DocumentPage } from './types'
import { getLastOcrError, ocrImageSource, terminateOcrWorker } from './ocr'

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.min.mjs',
  import.meta.url,
).toString()

const TEXT_LIKE = new Set([
  'text/plain',
  'text/markdown',
  'text/csv',
  'text/html',
  'text/css',
  'text/xml',
  'application/json',
  'application/xml',
  'application/javascript',
  'application/typescript',
])

const TEXT_EXT =
  /\.(txt|md|markdown|csv|json|xml|html|htm|css|js|jsx|ts|tsx|py|rb|go|rs|java|c|cpp|h|yml|yaml|toml|ini|log|rtf)$/i

const IMAGE_EXT = /\.(png|jpe?g|webp|gif|bmp)$/i
const PDF_TEXT_BATCH = 8

function yieldToUi(): Promise<void> {
  return new Promise((resolve) => {
    if (typeof requestAnimationFrame === 'function') {
      requestAnimationFrame(() => resolve())
    } else {
      setTimeout(resolve, 0)
    }
  })
}

export type ExtractProgress = {
  stage: string
  detail?: string
  progress?: number
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export interface ExtractResult {
  text: string
  pages: DocumentPage[]
  pageCount: number
  imageCount: number
}

async function readAsArrayBuffer(file: File): Promise<ArrayBuffer> {
  return file.arrayBuffer()
}

function guessTitleFromText(text: string, pageNum: number): string {
  const lines = text
    .split(/\n+/)
    .map((l) => l.replace(/\s+/g, ' ').trim())
    .filter(Boolean)

  for (const line of lines.slice(0, 8)) {
    if (line.length < 4 || line.length > 120) continue
    if (/^page\s*\d+/i.test(line)) continue
    if (/^\d+$/.test(line)) continue
    if (/^\[IMAGE/i.test(line)) continue
    if (/^[A-Z0-9]/.test(line) || /^[A-Z][^.!?]*$/.test(line)) {
      return line
    }
  }

  return lines[0]?.slice(0, 80) || `Page ${pageNum}`
}

function pagesToTaggedText(pages: DocumentPage[]): string {
  return pages
    .map((p) => `[[[PAGE ${p.page} | TITLE: ${p.title}]]]\n${p.text.trim()}`)
    .join('\n\n')
}

async function textFromPdfPage(page: pdfjs.PDFPageProxy): Promise<string> {
  const content = await page.getTextContent()

  type Item = { str: string; x: number; y: number }
  const items: Item[] = []
  for (const raw of content.items) {
    if (!('str' in raw) || !raw.str) continue
    const transform = 'transform' in raw ? (raw.transform as number[]) : null
    items.push({
      str: raw.str,
      x: transform ? transform[4] : 0,
      y: transform ? transform[5] : 0,
    })
  }

  items.sort((a, b) => b.y - a.y || a.x - b.x)

  const lines: string[] = []
  let currentY = Number.NaN
  let current = ''
  for (const item of items) {
    if (!Number.isFinite(currentY) || Math.abs(item.y - currentY) > 4) {
      if (current.trim()) lines.push(current.trim())
      current = item.str
      currentY = item.y
    } else {
      current += (current.endsWith(' ') || item.str.startsWith(' ') ? '' : ' ') + item.str
    }
  }
  if (current.trim()) lines.push(current.trim())

  return lines.join('\n').replace(/[ \t]+\n/g, '\n').trim()
}

async function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined
  try {
    return await Promise.race([
      promise,
      new Promise<T>((_, reject) => {
        timer = setTimeout(() => reject(new Error(`${label} timed out after ${ms / 1000}s`)), ms)
      }),
    ])
  } finally {
    if (timer) clearTimeout(timer)
  }
}

async function extractPdf(
  file: File,
  onProgress?: (p: ExtractProgress) => void,
): Promise<ExtractResult> {
  const data = await readAsArrayBuffer(file)
  onProgress?.({ stage: 'Opening PDF', detail: file.name, progress: 0.05 })
  const pdf = await pdfjs.getDocument({
    data,
  }).promise
  const pages: DocumentPage[] = new Array(pdf.numPages)

  for (let start = 1; start <= pdf.numPages; start += PDF_TEXT_BATCH) {
    const end = Math.min(start + PDF_TEXT_BATCH - 1, pdf.numPages)
    onProgress?.({
      stage: 'Reading PDF text',
      detail: `Pages ${start}–${end} of ${pdf.numPages}`,
      progress: 0.08 + (0.82 * end) / pdf.numPages,
    })

    await Promise.all(
      Array.from({ length: end - start + 1 }, async (_, idx) => {
        const i = start + idx
        const page = await pdf.getPage(i)
        const text = await textFromPdfPage(page)
        pages[i - 1] = {
          page: i,
          title: guessTitleFromText(text, i),
          text,
        }
      }),
    )
    await yieldToUi()
  }

  const tagged = pagesToTaggedText(pages)
  const body = tagged.replace(/\[\[\[PAGE[\s\S]*?\]\]\]/g, '').trim()
  onProgress?.({
    stage: 'PDF ready',
    detail: `Loaded ${pages.length} pages`,
    progress: 1,
  })

  return {
    pages,
    pageCount: pages.length,
    text: body ? tagged : pagesToTaggedText(pages.map((p) => ({ ...p, text: p.text || '(No text layer on this page)' }))),
    imageCount: 0,
  }
}

async function extractImageFile(
  file: File,
  onProgress?: (p: ExtractProgress) => void,
): Promise<ExtractResult> {
  onProgress?.({ stage: 'Reading image', detail: 'Starting OCR…', progress: 0.15 })
  let text = ''
  try {
    onProgress?.({ stage: 'Reading image', detail: file.name, progress: 0.45 })
    text = await withTimeout(ocrImageSource(file), 12_000, 'Image OCR')
  } catch (err) {
    void terminateOcrWorker()
    const detail = err instanceof Error ? err.message : getLastOcrError() || 'unknown error'
    throw new Error(`Couldn’t OCR “${file.name}”. ${detail}`)
  }
  onProgress?.({ stage: 'Reading image', detail: 'OCR complete', progress: 1 })
  if (!text.trim()) {
    throw new Error(
      `No readable text found in “${file.name}”. Use a clearer image with visible text.`,
    )
  }
  const pages: DocumentPage[] = [
    {
      page: 1,
      title: file.name.replace(/\.[^.]+$/, '') || 'Image',
      text: `[IMAGE OCR]\n${text}`,
    },
  ]
  return {
    pages,
    pageCount: 1,
    text: pagesToTaggedText(pages),
    imageCount: 1,
  }
}

async function extractDocx(
  file: File,
  onProgress?: (p: ExtractProgress) => void,
): Promise<ExtractResult> {
  const data = await readAsArrayBuffer(file)
  onProgress?.({ stage: 'Reading DOCX', detail: 'Extracting text…', progress: 0.35 })

  const result = await mammoth.extractRawText({ arrayBuffer: data })
  const raw = result.value.trim()
  // Skip heavy DOCX image OCR so uploads open quickly.
  const imageCount = 0

  onProgress?.({ stage: 'Reading DOCX', detail: 'Building pages…', progress: 0.7 })

  const chunks = raw.split(/\n{2,}/).filter(Boolean)
  const pages: DocumentPage[] = []
  let buf = ''
  let page = 1

  for (const chunk of chunks) {
    if ((buf + '\n\n' + chunk).length > 2800 && buf) {
      pages.push({
        page,
        title: guessTitleFromText(buf, page),
        text: buf.trim(),
      })
      page += 1
      buf = chunk
    } else {
      buf = buf ? `${buf}\n\n${chunk}` : chunk
    }
  }
  if (buf.trim()) {
    pages.push({
      page,
      title: guessTitleFromText(buf, page),
      text: buf.trim(),
    })
  }

  if (!pages.length) {
    pages.push({ page: 1, title: file.name, text: raw })
  }

  onProgress?.({ stage: 'Reading DOCX', detail: 'Done', progress: 1 })

  return {
    pages,
    pageCount: pages.length,
    text: pagesToTaggedText(pages),
    imageCount,
  }
}

async function extractTextFile(file: File): Promise<ExtractResult> {
  const raw = (await file.text()).trim()
  const pages: DocumentPage[] = [
    {
      page: 1,
      title: file.name.replace(/\.[^.]+$/, '') || 'Document',
      text: raw,
    },
  ]
  return {
    pages,
    pageCount: 1,
    text: pagesToTaggedText(pages),
    imageCount: 0,
  }
}

function isDocx(file: File): boolean {
  return (
    file.type ===
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document' ||
    /\.docx$/i.test(file.name)
  )
}

function isPdf(file: File): boolean {
  return file.type === 'application/pdf' || /\.pdf$/i.test(file.name)
}

export function isImageFile(file: File): boolean {
  return file.type.startsWith('image/') || IMAGE_EXT.test(file.name)
}

function isLegacyDoc(file: File): boolean {
  return (
    file.type === 'application/msword' ||
    (/\.doc$/i.test(file.name) && !/\.docx$/i.test(file.name))
  )
}

function isTextLike(file: File): boolean {
  return TEXT_LIKE.has(file.type) || TEXT_EXT.test(file.name) || file.type.startsWith('text/')
}

export async function extractFileContent(
  file: File,
  onProgress?: (p: ExtractProgress) => void,
): Promise<ExtractResult> {
  if (isPdf(file)) return extractPdf(file, onProgress)
  if (isImageFile(file)) return extractImageFile(file, onProgress)
  if (isDocx(file)) return extractDocx(file, onProgress)

  if (isLegacyDoc(file)) {
    throw new Error(
      'Legacy .doc files aren’t supported. Please save as .docx or PDF and try again.',
    )
  }

  if (isTextLike(file)) return extractTextFile(file)

  try {
    const result = await extractTextFile(file)
    if (result.text.trim().length > 0 && !result.text.includes('\u0000')) {
      return result
    }
  } catch {
    // fall through
  }

  throw new Error(
    `Couldn’t extract content from “${file.name}”. Try PDF, DOCX, PNG, JPG, WEBP, TXT, MD, CSV, or JSON.`,
  )
}

/** @deprecated use extractFileContent */
export async function extractFileText(file: File): Promise<string> {
  const result = await extractFileContent(file)
  return result.text
}

export function summarizeExtract(text: string): { wordCount: number; preview: string } {
  const plain = text.replace(/\[\[\[PAGE[\s\S]*?\]\]\]/g, ' ')
  const words = plain.trim().split(/\s+/).filter(Boolean)
  const preview = plain.trim().slice(0, 420).replace(/\s+/g, ' ')
  return {
    wordCount: words.length,
    preview: preview + (plain.trim().length > 420 ? '…' : ''),
  }
}
