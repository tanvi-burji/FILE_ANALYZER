import {
  useEffect,
  useId,
  useRef,
  useState,
  type DragEvent,
  type FormEvent,
} from 'react'
import { DocumentPreview } from './components/DocumentPreview'
import { answerQuestion, indexDocument } from './lib/ask'
import { renderAnswerHtml } from './lib/citations'
import {
  extractFileContent,
  formatBytes,
  isImageFile,
  summarizeExtract,
  type ExtractProgress,
} from './lib/extract'
import { setStoredCredentials } from './lib/settings'
import type { AnalyzedFile, ChatMessage, DocumentBrain } from './lib/types'

const SUGGESTIONS = [
  'Summarize this document',
  'What is CDF / Cognite Data Fusion in this file?',
  'What does the timeline or chart show?',
  'List important dates',
]

function Logo({ size = 40 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" fill="none" aria-hidden>
      <rect width="64" height="64" rx="16" fill="#0A0614" />
      <rect x="4" y="4" width="56" height="56" rx="13" stroke="url(#g)" strokeWidth="2" />
      <path
        d="M16 24h32M16 33h24M16 42h16"
        stroke="#7BA7FF"
        strokeWidth="3.2"
        strokeLinecap="round"
      />
      <circle cx="44" cy="42" r="8" stroke="#B48CFF" strokeWidth="2.8" />
      <path d="M49.5 47.5L55 53" stroke="#B48CFF" strokeWidth="2.8" strokeLinecap="round" />
      <defs>
        <linearGradient id="g" x1="8" y1="8" x2="56" y2="56">
          <stop stopColor="#5B8CFF" />
          <stop offset="1" stopColor="#9B6BFF" />
        </linearGradient>
      </defs>
    </svg>
  )
}

function uid() {
  return crypto.randomUUID()
}

function isPdfName(name: string, type?: string) {
  return type === 'application/pdf' || /\.pdf$/i.test(name)
}

export default function App() {
  const inputId = useId()
  const fileInputRef = useRef<HTMLInputElement>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const [dragging, setDragging] = useState(false)
  const [busy, setBusy] = useState(false)
  const [indexing, setIndexing] = useState(false)
  const [thinking, setThinking] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [extractStatus, setExtractStatus] = useState<ExtractProgress | null>(null)
  const [file, setFile] = useState<AnalyzedFile | null>(null)
  const [fileData, setFileData] = useState<ArrayBuffer | null>(null)
  const [isPdf, setIsPdf] = useState(false)
  const [isImage, setIsImage] = useState(false)
  const [brain, setBrain] = useState<DocumentBrain | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [question, setQuestion] = useState('')
  const [currentPage, setCurrentPage] = useState(1)
  const [flashToken, setFlashToken] = useState(0)

  const hasFile = Boolean(file && file.status !== 'error')

  useEffect(() => {
    setStoredCredentials('browser', '')
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, thinking, indexing])

  function jumpToPage(page: number) {
    if (!file) return
    const safe = Math.min(Math.max(page, 1), Math.max(file.pageCount, 1))
    setCurrentPage(safe)
    setFlashToken((n) => n + 1)
  }

  async function processFile(list: FileList | File[]) {
    const next = Array.from(list)[0]
    if (!next) return

    setError(null)
    setBusy(true)
    setMessages([])
    setBrain(null)
    setCurrentPage(1)
    setFileData(null)
    setFile(null)
    setExtractStatus({ stage: 'Opening file', detail: next.name, progress: 0.02 })

    const pdf = isPdfName(next.name, next.type)
    const image = isImageFile(next)
    setIsPdf(pdf)
    setIsImage(image)

    try {
      const buffer = await next.arrayBuffer()
      setFileData(buffer.slice(0))

      const fileId = uid()
      setFile({
        id: fileId,
        name: next.name,
        type: next.type || 'unknown',
        size: next.size,
        text: '',
        pages: [],
        pageCount: 1,
        wordCount: 0,
        preview: '',
        imageCount: 0,
        status: 'extracting',
      })
      setBusy(false)

      const extracted = await extractFileContent(next, setExtractStatus)
      if (!extracted.text.trim()) throw new Error('No readable text or image content found.')

      const { wordCount, preview } = summarizeExtract(extracted.text)
      const analyzed: AnalyzedFile = {
        id: fileId,
        name: next.name,
        type: next.type || 'unknown',
        size: next.size,
        text: extracted.text,
        pages: extracted.pages,
        pageCount: extracted.pageCount,
        wordCount,
        preview,
        imageCount: extracted.imageCount,
        status: 'ready',
      }

      const indexed = await indexDocument(analyzed)
      setBrain(indexed)
      setFile(analyzed)
      setExtractStatus(null)

      setMessages([
        {
          id: uid(),
          role: 'assistant',
          content: `Ready. **${analyzed.name}** is loaded (${extracted.pageCount} page${extracted.pageCount === 1 ? '' : 's'}, ${wordCount.toLocaleString()} words).\n\n${indexed.brief.summary}\n\nAnswers quote the file and link to the page. Ask about definitions, dates, or timelines.`,
          sources: [],
          sourceRefs: [],
        },
      ])
    } catch (err) {
      console.error('ScanAsk processFile failed', err)
      const message = err instanceof Error ? err.message : 'Could not read this file.'
      setError(message)
      setFile(null)
      setFileData(null)
      setBusy(false)
    } finally {
      setIndexing(false)
      setExtractStatus(null)
    }
  }

  function onDrop(e: DragEvent) {
    e.preventDefault()
    setDragging(false)
    if (e.dataTransfer.files?.length) void processFile(e.dataTransfer.files)
  }

  async function ask(q: string) {
    const trimmed = q.trim()
    if (!trimmed || thinking || indexing || !file || file.status !== 'ready') return

    setQuestion('')
    setError(null)
    const userMsg: ChatMessage = { id: uid(), role: 'user', content: trimmed }
    const history = [...messages, userMsg]
    setMessages(history)
    setThinking(true)

    try {
      const { answer, sources, sourceRefs } = await answerQuestion(trimmed, [file], brain, history)
      if (sourceRefs[0]?.page) jumpToPage(sourceRefs[0].page)
      setMessages((m) => [
        ...m,
        { id: uid(), role: 'assistant', content: answer, sources, sourceRefs },
      ])
    } catch (err) {
      setMessages((m) => [
        ...m,
        {
          id: uid(),
          role: 'assistant',
          content: err instanceof Error ? err.message : 'Something went wrong. Try again.',
        },
      ])
    } finally {
      setThinking(false)
    }
  }

  function reset() {
    setFile(null)
    setFileData(null)
    setIsPdf(false)
    setIsImage(false)
    setBrain(null)
    setMessages([])
    setQuestion('')
    setError(null)
    setIndexing(false)
    setBusy(false)
    setCurrentPage(1)
    setFlashToken(0)
    setExtractStatus(null)
  }

  const accept =
    '.pdf,.doc,.docx,.png,.jpg,.jpeg,.webp,.gif,.bmp,.txt,.md,.csv,.json,.xml,.html,.htm,.log,.yml,.yaml,.rtf,application/pdf,image/*,application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document,text/*'

  return (
    <div className="app">
      <div className="atmosphere" aria-hidden />
      <div className={`shell${hasFile ? ' workspace' : ' landing'}`}>
        <header className="topbar box">
          <button type="button" className="brand" onClick={reset}>
            <Logo />
            <span className="wordmark">
              Scan<span>Ask</span>
            </span>
          </button>
          {hasFile ? (
            <button type="button" className="btn btn-ghost" onClick={reset}>
              New file
            </button>
          ) : null}
        </header>

        {!hasFile ? (
          <section className="hero-stack">
            <div className="box hero-copy">
            <p className="eyebrow">Quoted answers · small page refs</p>
              <h1>Ask your file</h1>
              <p className="lede">
                Upload a PDF, DOCX, or text file. ScanAsk extracts the text and answers with
                quotes from the file — including the page to open.
              </p>
            </div>

            <div className="feature-row">
              <div className="box feature">
                <strong>Fast</strong>
                <span>Opens on the text layer — no model download wait</span>
              </div>
              <div className="box feature">
                <strong>Accurate</strong>
                <span>Answers quote the file; names stay as written</span>
              </div>
              <div className="box feature">
                <strong>Cited</strong>
                <span>Every answer links to the page in preview</span>
              </div>
            </div>

            <div
              className={`box upload-zone${dragging ? ' dragging' : ''}${busy ? ' busy' : ''}`}
              onDragEnter={(e) => {
                e.preventDefault()
                setDragging(true)
              }}
              onDragOver={(e) => e.preventDefault()}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
            >
              <div className="upload-icon" aria-hidden>
                <svg width="44" height="44" viewBox="0 0 44 44" fill="none">
                  <rect
                    x="8"
                    y="6"
                    width="28"
                    height="32"
                    rx="6"
                    stroke="#7BA7FF"
                    strokeWidth="2"
                  />
                  <path
                    d="M15 16h14M15 22h14M15 28h9"
                    stroke="#B48CFF"
                    strokeWidth="2"
                    strokeLinecap="round"
                  />
                  <circle cx="30" cy="30" r="7" fill="#120A22" stroke="#9B6BFF" strokeWidth="2" />
                </svg>
              </div>
              <h2>{busy ? 'Reading your file…' : 'Drop your file here'}</h2>
              <p>PDF · PNG · JPG · WEBP · DOCX · TXT</p>
              {extractStatus && (
                <div className="extract-status">
                  <strong>{extractStatus.stage}</strong>
                  {extractStatus.detail && <span>{extractStatus.detail}</span>}
                  {typeof extractStatus.progress === 'number' && (
                    <div className="mini-bar">
                      <span style={{ width: `${Math.round(extractStatus.progress * 100)}%` }} />
                    </div>
                  )}
                </div>
              )}
              <button
                type="button"
                className="btn btn-primary"
                disabled={busy}
                onClick={() => fileInputRef.current?.click()}
              >
                {busy ? 'Opening…' : 'Choose file'}
              </button>
              <input
                ref={fileInputRef}
                id={inputId}
                className="hidden-input"
                type="file"
                accept={accept}
                onChange={(e) => {
                  if (e.target.files) void processFile(e.target.files)
                  e.target.value = ''
                }}
              />
            </div>

            {error && <div className="box error-banner">{error}</div>}
          </section>
        ) : (
          <section className="work-stack">
            <div className="box file-strip">
              <div>
                <strong>{file?.name}</strong>
                <span>
                  {file
                    ? `${formatBytes(file.size)} · ${file.pageCount} pages · ${file.wordCount.toLocaleString()} words${file.imageCount ? ` · ${file.imageCount} image page(s)` : ''}`
                    : ''}
                </span>
              </div>
              <div className={`status-pill${file?.status === 'ready' ? ' ok' : ''}`}>
                {file?.status === 'extracting'
                  ? extractStatus?.stage || 'Reading…'
                  : 'Ready'}
              </div>
            </div>

            {file?.status === 'extracting' && extractStatus && (
              <div className="box model-progress">
                <div className="model-progress-bar">
                  <span
                    style={{
                      width: `${Math.round((extractStatus.progress ?? 0.08) * 100)}%`,
                    }}
                  />
                </div>
                <p>
                  {extractStatus.stage}
                  {extractStatus.detail ? ` — ${extractStatus.detail}` : ''}
                </p>
              </div>
            )}

            <div className="split-grid">
              <DocumentPreview
                fileData={fileData}
                isPdf={isPdf}
                isImage={isImage}
                pages={file?.pages || []}
                pageCount={file?.pageCount || 1}
                currentPage={currentPage}
                onPageChange={jumpToPage}
                flashToken={flashToken}
              />

              <div className="box chat-panel">
                <div className="chat-header">
                  <h2>Ask about this file</h2>
                  <p>Quoted from the file — product names like CDF are copied exactly, not guessed.</p>
                </div>

                <div className="messages">
                  {(file?.status === 'extracting' || indexing) && messages.length === 0 && (
                    <div className="msg assistant box-msg">
                      <div className="thinking">
                        {file?.status === 'extracting'
                          ? extractStatus?.detail || 'Reading your file'
                          : 'Preparing your document'}
                        <span className="dots" aria-hidden>
                          <span />
                          <span />
                          <span />
                        </span>
                      </div>
                    </div>
                  )}

                  {messages.length === 0 && !indexing && file?.status !== 'extracting' && (
                    <div className="empty-chat box-msg">
                      <strong>Ask anything about this file</strong>
                      <div className="suggestions">
                        {SUGGESTIONS.map((s) => (
                          <button key={s} type="button" className="chip" onClick={() => void ask(s)}>
                            {s}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}

                  {messages.map((m) => (
                    <div key={m.id} className={`msg box-msg ${m.role}`}>
                      {m.role === 'assistant' ? (
                        <>
                          <div
                            className="answer-body"
                            dangerouslySetInnerHTML={{ __html: renderAnswerHtml(m.content) }}
                          />
                          {m.sourceRefs && m.sourceRefs.length > 0 && (
                            <div className="sources-block">
                              <span className="sources-label">Refs</span>
                              <div className="source-chips">
                                {m.sourceRefs.map((ref) => (
                                  <button
                                    key={ref.label}
                                    type="button"
                                    className="source-chip"
                                    title={ref.title}
                                    onClick={() => jumpToPage(ref.page)}
                                  >
                                    p.{ref.page}
                                  </button>
                                ))}
                              </div>
                            </div>
                          )}
                        </>
                      ) : (
                        <div>{m.content}</div>
                      )}
                    </div>
                  ))}

                  {thinking && (
                    <div className="msg assistant box-msg">
                      <div className="thinking">
                        Finding the answer in your file
                        <span className="dots" aria-hidden>
                          <span />
                          <span />
                          <span />
                        </span>
                      </div>
                    </div>
                  )}
                  <div ref={bottomRef} />
                </div>

                <form
                  className="composer box-inner"
                  onSubmit={(e: FormEvent) => {
                    e.preventDefault()
                    void ask(question)
                  }}
                >
                  <textarea
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    placeholder={
                      file?.status === 'extracting'
                        ? 'Reading file…'
                        : indexing
                          ? 'Preparing document…'
                          : 'Ask a question about text or images…'
                    }
                    rows={1}
                    disabled={indexing || file?.status !== 'ready'}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault()
                        void ask(question)
                      }
                    }}
                  />
                  <button
                    type="submit"
                    className="btn btn-primary"
                    disabled={thinking || indexing || !question.trim()}
                  >
                    Ask
                  </button>
                </form>
              </div>
            </div>

            {error && <div className="box error-banner">{error}</div>}
          </section>
        )}
      </div>
    </div>
  )
}
