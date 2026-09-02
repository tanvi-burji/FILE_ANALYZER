import { useEffect, useMemo, useRef, useState } from 'react'
import * as pdfjs from 'pdfjs-dist'
import type { PDFDocumentProxy } from 'pdfjs-dist'
import type { DocumentPage } from '../lib/types'

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.min.mjs',
  import.meta.url,
).toString()

type Props = {
  fileData: ArrayBuffer | null
  isPdf: boolean
  isImage?: boolean
  pages: DocumentPage[]
  pageCount: number
  currentPage: number
  onPageChange: (page: number) => void
  flashToken?: number
}

function cleanPreviewText(raw: string): string {
  return raw
    .replace(/\[\[\[PAGE[\s\S]*?\]\]\]\s*/g, '')
    .replace(/\[(?:PAGE )?IMAGE[^\]]*\]\s*/gi, '')
    .replace(/\[EMBEDDED IMAGE[^\]]*\]\s*/gi, '')
    .replace(/\[DOCX IMAGE[^\]]*\]\s*/gi, '')
    .replace(/\n{3,}/g, '\n\n')
    .trim()
}

export function DocumentPreview({
  fileData,
  isPdf,
  isImage = false,
  pages,
  pageCount,
  currentPage,
  onPageChange,
  flashToken = 0,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const stageRef = useRef<HTMLDivElement>(null)
  const panelRef = useRef<HTMLDivElement>(null)
  const pdfRef = useRef<PDFDocumentProxy | null>(null)
  const [docReady, setDocReady] = useState(0)
  const [pdfPageCount, setPdfPageCount] = useState(0)
  const [loading, setLoading] = useState(false)
  const [flash, setFlash] = useState(false)
  const [renderError, setRenderError] = useState<string | null>(null)
  const [showText, setShowText] = useState(false)

  const textPage = pages.find((p) => p.page === currentPage) || pages[0]
  const safeCount = Math.max(pdfPageCount, pageCount, pages.length, 1)
  const previewText = cleanPreviewText(textPage?.text || '')

  const imageUrl = useMemo(() => {
    if (!isImage || !fileData) return null
    const blob = new Blob([new Uint8Array(fileData)])
    return URL.createObjectURL(blob)
  }, [fileData, isImage])

  useEffect(() => {
    return () => {
      if (imageUrl) URL.revokeObjectURL(imageUrl)
    }
  }, [imageUrl])

  useEffect(() => {
    if (!flashToken) return
    setFlash(true)
    panelRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
    const t = window.setTimeout(() => setFlash(false), 1100)
    return () => window.clearTimeout(t)
  }, [flashToken, currentPage])

  useEffect(() => {
    let cancelled = false

    async function load() {
      if (pdfRef.current) {
        try {
          await pdfRef.current.cleanup()
        } catch {
          // ignore
        }
        pdfRef.current = null
      }

      if (!isPdf || !fileData) {
        setDocReady(0)
        setPdfPageCount(0)
        return
      }

      setLoading(true)
      setRenderError(null)
      try {
        const data = fileData.slice(0)
        const pdf = await pdfjs.getDocument({ data }).promise
        if (cancelled) {
          await pdf.cleanup()
          return
        }
        pdfRef.current = pdf
        setPdfPageCount(pdf.numPages)
        setDocReady((n) => n + 1)
      } catch (err) {
        if (!cancelled) {
          setRenderError(
            err instanceof Error ? err.message : 'Could not open PDF preview.',
          )
        }
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    void load()
    return () => {
      cancelled = true
    }
  }, [fileData, isPdf])

  useEffect(() => {
    let cancelled = false

    async function render() {
      const pdf = pdfRef.current
      const canvas = canvasRef.current
      if (!isPdf || !pdf || !canvas || !docReady) return

      setLoading(true)
      try {
        const pageNum = Math.min(Math.max(currentPage, 1), pdf.numPages)
        const page = await pdf.getPage(pageNum)
        if (cancelled) return

        const base = page.getViewport({ scale: 1 })
        const maxWidth = Math.max(stageRef.current?.clientWidth || 440, 280)
        const scale = Math.min(2, (maxWidth - 16) / base.width)
        const viewport = page.getViewport({ scale })
        const context = canvas.getContext('2d')
        if (!context) return

        canvas.width = viewport.width
        canvas.height = viewport.height
        context.fillStyle = '#ffffff'
        context.fillRect(0, 0, canvas.width, canvas.height)

        await page.render({
          canvasContext: context,
          canvas,
          viewport,
          background: '#ffffff',
        }).promise

        if (!cancelled) setRenderError(null)
      } catch (err) {
        if (!cancelled) {
          setRenderError(
            err instanceof Error ? err.message : 'Could not render this page.',
          )
        }
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    void render()
    return () => {
      cancelled = true
    }
  }, [currentPage, isPdf, docReady])

  useEffect(() => {
    return () => {
      if (pdfRef.current) {
        void pdfRef.current.cleanup()
        pdfRef.current = null
      }
    }
  }, [])

  const showPdfPage = isPdf && fileData && !renderError
  const showImage = isImage && imageUrl
  const showTextFallback = !showPdfPage && !showImage

  return (
    <aside ref={panelRef} className={`box preview-panel${flash ? ' flash' : ''}`}>
      <div className="preview-head">
        <div>
          <p className="preview-kicker">File preview</p>
          <h3>{isImage ? 'Uploaded image' : `Page ${currentPage}`}</h3>
        </div>
        {!isImage && (
          <div className="preview-toolbar">
            <button
              type="button"
              className="nav-btn"
              disabled={currentPage <= 1}
              onClick={() => onPageChange(Math.max(1, currentPage - 1))}
              aria-label="Previous page"
            >
              ‹
            </button>
            <span className="page-fraction">
              {currentPage} / {safeCount}
            </span>
            <button
              type="button"
              className="nav-btn"
              disabled={currentPage >= safeCount}
              onClick={() => onPageChange(Math.min(safeCount, currentPage + 1))}
              aria-label="Next page"
            >
              ›
            </button>
          </div>
        )}
      </div>

      <div className="preview-stage" ref={stageRef}>
        {loading && <div className="preview-loading">Loading page…</div>}

        {showImage && (
          <div className="page-frame">
            <img src={imageUrl!} alt="Uploaded preview" className="image-preview" />
          </div>
        )}

        {showPdfPage && (
          <div className="page-frame">
            <canvas ref={canvasRef} className="pdf-canvas" />
          </div>
        )}

        {showTextFallback && (
          <div className="page-frame text-fallback">
            <p className="text-fallback-label">Page {currentPage}</p>
            <div className="text-fallback-body">
              {previewText || 'No content for this page.'}
            </div>
          </div>
        )}

        {renderError && <div className="preview-note">{renderError}</div>}
      </div>

      {!isImage && (
        <div className="page-rail" aria-label="Jump to page">
          {Array.from({ length: Math.min(safeCount, 80) }, (_, i) => i + 1).map((p) => (
            <button
              key={p}
              type="button"
              className={`page-dot${p === currentPage ? ' active' : ''}`}
              onClick={() => onPageChange(p)}
            >
              {p}
            </button>
          ))}
        </div>
      )}

      {(showPdfPage || showImage) && (
        <div className="preview-extra">
          <button
            type="button"
            className="text-toggle"
            onClick={() => setShowText((v) => !v)}
          >
            {showText ? 'Hide extracted text' : 'Show extracted text'}
          </button>
          {showText && (
            <div className="text-page-card">
              <pre>{previewText || 'No extracted text for this page.'}</pre>
            </div>
          )}
        </div>
      )}
    </aside>
  )
}
