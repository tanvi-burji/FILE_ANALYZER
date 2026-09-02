export const BROWSER_MODEL_ID = 'Llama-3.2-3B-Instruct-q4f16_1-MLC'

/** Hard limit for this WebLLM build (prompt + completion). */
export const BROWSER_CONTEXT_TOKENS = 4096
/** Leave headroom for the reply. */
export const BROWSER_MAX_OUTPUT_TOKENS = 480
/** Rough char budget for system + user messages combined. */
export const BROWSER_MAX_INPUT_CHARS = 10_500

export type BrowserLoadProgress = {
  progress: number
  text: string
}

type ProgressListener = (report: BrowserLoadProgress) => void
type Engine = {
  chat: {
    completions: {
      create: (req: {
        messages: { role: 'system' | 'user' | 'assistant'; content: string }[]
        temperature?: number
        max_tokens?: number
      }) => Promise<{ choices?: { message?: { content?: string | null } }[] }>
    }
  }
}

let enginePromise: Promise<Engine> | null = null
let engine: Engine | null = null
const listeners = new Set<ProgressListener>()

export function onBrowserLlmProgress(listener: ProgressListener): () => void {
  listeners.add(listener)
  return () => listeners.delete(listener)
}

function emit(progress: number, text: string) {
  const payload = { progress, text }
  for (const listener of listeners) listener(payload)
}

export function isBrowserLlmReady(): boolean {
  return Boolean(engine)
}

export function hasWebGpu(): boolean {
  return typeof navigator !== 'undefined' && 'gpu' in navigator
}

/** Conservative token estimate for English + markup. */
export function estimateTokens(text: string): number {
  return Math.ceil(text.length / 3.2)
}

function trimToTokenBudget(text: string, maxTokens: number): string {
  const maxChars = Math.floor(maxTokens * 3.2)
  if (text.length <= maxChars) return text
  return (
    text.slice(0, Math.floor(maxChars * 0.72)) +
    '\n\n[…truncated to fit model context…]\n\n' +
    text.slice(-Math.floor(maxChars * 0.22))
  )
}

export async function getBrowserEngine(): Promise<Engine> {
  if (engine) return engine

  if (!hasWebGpu()) {
    throw new Error(
      'This browser does not support WebGPU. Use Chrome or Edge (latest), then reload ScanAsk to run the local model.',
    )
  }

  if (!enginePromise) {
    enginePromise = (async () => {
      const { CreateMLCEngine } = await import('@mlc-ai/web-llm')
      // Keep default 4096 context for this model build; prompts are sized to fit.
      const created = await CreateMLCEngine(BROWSER_MODEL_ID, {
        initProgressCallback: (report) => {
          const raw = report.text || 'Loading browser model…'
          const text = /cache|fetch/i.test(raw)
            ? `${raw} First visit only — later questions use the cached model.`
            : raw
          emit(report.progress, text)
        },
      })
      engine = created
      emit(1, 'Browser model ready')
      return created
    })().catch((err) => {
      enginePromise = null
      throw err instanceof Error
        ? err
        : new Error('Failed to load the browser LLM. Check WebGPU and try again.')
    })
  }

  return enginePromise
}

export async function browserChat(system: string, user: string): Promise<string> {
  const mlc = await getBrowserEngine()

  const systemTrimmed = trimToTokenBudget(system, 220)
  const reserved =
    estimateTokens(systemTrimmed) + BROWSER_MAX_OUTPUT_TOKENS + 48
  const userBudget = Math.max(900, BROWSER_CONTEXT_TOKENS - reserved)
  const userTrimmed = trimToTokenBudget(user, userBudget)

  const reply = await mlc.chat.completions.create({
    messages: [
      { role: 'system', content: systemTrimmed },
      { role: 'user', content: userTrimmed },
    ],
    temperature: 0,
    max_tokens: BROWSER_MAX_OUTPUT_TOKENS,
  })

  const text = reply.choices?.[0]?.message?.content
  const content = typeof text === 'string' ? text.trim() : ''
  if (!content) throw new Error('The browser model returned an empty response.')
  return content
}
