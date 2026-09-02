import { browserChat, BROWSER_MODEL_ID } from './browserLlm'
import type { ProviderId } from './settings'
import type { AnalyzedFile, DocumentBrain, KnowledgeBrief } from './types'

/** Context budgets by provider. Browser models need smaller chunks. */
const CHUNK_CHARS: Record<ProviderId, number> = {
  browser: 4_500,
  openrouter: 80_000,
  groq: 10_000,
  gemini: 80_000,
  openai: 60_000,
}

const ASK_CONTEXT_CHARS: Record<ProviderId, number> = {
  // Must stay well under WebLLM's 4096-token context (≈ prompt + reply)
  browser: 7_200,
  openrouter: 280_000,
  groq: 9_000,
  gemini: 280_000,
  openai: 80_000,
}

/** Max document sample size for a single Groq training call. */
const GROQ_TRAIN_SAMPLE_CHARS = 11_000

const LARGE_TRAIN_CHARS = 300_000

const OPENROUTER_DEFAULT_MODEL = 'google/gemini-2.0-flash-001'

const STOP = new Set([
  'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
  'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
  'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
  'can', 'this', 'that', 'these', 'those', 'it', 'its', 'as', 'if', 'then', 'than',
  'so', 'not', 'no', 'yes', 'what', 'which', 'who', 'whom', 'when', 'where', 'why',
  'how', 'i', 'you', 'we', 'they', 'he', 'she', 'me', 'my', 'your', 'our', 'about',
  'please', 'tell', 'explain', 'describe', 'give', 'show', 'find', 'list', 'summarize',
])

export type LlmProvider = ProviderId

export interface LlmConfig {
  provider: LlmProvider
  apiKey: string
  model?: string
}

type DocPiece = {
  file: string
  text: string
  page?: number
  title?: string
}

function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms))
}

function chunkText(text: string, size: number, overlap = 400): string[] {
  const clean = text.replace(/\r\n/g, '\n').trim()
  if (!clean) return []
  if (clean.length <= size) return [clean]

  const chunks: string[] = []
  let i = 0
  while (i < clean.length) {
    const end = Math.min(i + size, clean.length)
    let slice = clean.slice(i, end)
    if (end < clean.length) {
      const lastBreak = Math.max(
        slice.lastIndexOf('\n\n'),
        slice.lastIndexOf('\n'),
        slice.lastIndexOf('. '),
        slice.lastIndexOf(' '),
      )
      if (lastBreak > size * 0.4) slice = slice.slice(0, lastBreak + 1)
    }
    chunks.push(slice.trim())
    i += Math.max(slice.length - overlap, 1)
  }
  return chunks.filter(Boolean)
}

function buildPiecesFromFiles(files: AnalyzedFile[], chunkSize: number): DocPiece[] {
  const pieces: DocPiece[] = []

  for (const file of files) {
    if (file.pages?.length) {
      for (const page of file.pages) {
        const header = `[[[PAGE ${page.page} | TITLE: ${page.title}]]]\n`
        if (header.length + page.text.length <= chunkSize) {
          pieces.push({
            file: file.name,
            text: header + page.text,
            page: page.page,
            title: page.title,
          })
        } else {
          for (const part of chunkText(page.text, Math.max(chunkSize - 100, 800))) {
            pieces.push({
              file: file.name,
              text: header + part,
              page: page.page,
              title: page.title,
            })
          }
        }
      }
    } else {
      for (const part of chunkText(file.text, chunkSize)) {
        pieces.push({ file: file.name, text: part })
      }
    }
  }

  return pieces
}

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s'-]/g, ' ')
    .split(/\s+/)
    .filter((t) => t.length > 1 && !STOP.has(t))
}

function stem(token: string): string {
  return token
    .replace(/(ing|ed|ly|tion|sion|ments|ment|ness|ers|er|ous|ies|ied|es|s)$/i, '')
    .replace(/i$/, 'y')
}

function collectAcronyms(files: AnalyzedFile[]): Map<string, string> {
  const map = new Map<string, string>()
  const text = files.map((f) => f.text).join('\n')
  const compact = /\b([A-Z]{2,10})\s*\(([^)]{5,90})\)/g
  const expandedFirst = /\b([A-Z][A-Za-z0-9+/&.-]*(?:\s+[A-Z][A-Za-z0-9+/&.-]*){1,6})\s*\(([A-Z]{2,10})\)/g
  let m: RegExpExecArray | null
  while ((m = compact.exec(text)) !== null) {
    map.set(m[1].toLowerCase(), m[2].replace(/\s+/g, ' ').trim())
  }
  while ((m = expandedFirst.exec(text)) !== null) {
    map.set(m[2].toLowerCase(), m[1].replace(/\s+/g, ' ').trim())
  }
  // Cognite files: CDF is Cognite Data Fusion unless the file defines it otherwise as Fusion.
  if (/cognite data fusion/i.test(text)) {
    const current = map.get('cdf') || ''
    if (!current || /fabric/i.test(current) || /fusion/i.test(current)) {
      map.set('cdf', 'Cognite Data Fusion')
    }
  }
  return map
}

function extraTokensFromDocument(question: string, files: AnalyzedFile[]): string[] {
  const q = question.toLowerCase()
  const extras: string[] = []
  for (const [acro, expansion] of collectAcronyms(files)) {
    if (q.includes(acro) || q.includes(expansion.toLowerCase())) {
      extras.push(acro, ...tokenize(expansion))
    }
  }
  return extras
}

function expandQueryTokens(question: string, extra: string[] = []): string[] {
  const base = tokenize(`${question} ${extra.join(' ')}`)
  const expanded = new Set<string>(base)

  const synonyms: Record<string, string[]> = {
    deadline: ['due', 'date', 'timeline', 'schedule'],
    due: ['deadline', 'date'],
    salary: ['compensation', 'pay', 'wage', 'ctc', 'remuneration'],
    pay: ['salary', 'compensation', 'wage'],
    cost: ['price', 'fee', 'amount', 'expense'],
    price: ['cost', 'fee', 'amount'],
    people: ['person', 'persons', 'employee', 'staff', 'team', 'member'],
    who: ['person', 'name', 'author', 'employee'],
    when: ['date', 'time', 'deadline', 'schedule'],
    where: ['location', 'place', 'address', 'venue'],
    why: ['reason', 'because', 'purpose'],
    how: ['method', 'process', 'steps', 'procedure'],
    summary: ['overview', 'abstract', 'conclusion'],
    risk: ['issue', 'problem', 'concern', 'hazard'],
    requirement: ['must', 'shall', 'need', 'obligation'],
    obligation: ['requirement', 'duty', 'shall', 'must'],
    image: ['figure', 'photo', 'picture', 'diagram', 'chart', 'scan', 'screenshot', 'timeline', 'graph'],
    figure: ['image', 'diagram', 'chart', 'photo', 'graph'],
    chart: ['graph', 'figure', 'diagram', 'table', 'timeline'],
    graph: ['chart', 'figure', 'diagram', 'timeline'],
    timeline: ['roadmap', 'gantt', 'milestone', 'schedule', 'phase', 'quarter', 'chart', 'graph'],
    roadmap: ['timeline', 'plan', 'milestone', 'phase', 'schedule'],
    table: ['chart', 'grid', 'rows'],
    define: ['definition', 'means', 'meaning'],
    meaning: ['definition', 'means', 'define'],
    scope: ['sow', 'services', 'deliverables', 'work', 'engagement', 'duties'],
    sow: ['scope', 'services', 'deliverables', 'work'],
    services: ['scope', 'sow', 'deliverables', 'work', 'performance'],
    deliverables: ['scope', 'services', 'outputs', 'work'],
    work: ['scope', 'services', 'sow', 'duties', 'performance'],
    agreement: ['contract', 'msa', 'master'],
    consultant: ['contractor', 'provider', 'vendor', 'party'],
  }

  for (const token of base) {
    expanded.add(stem(token))
    const syns = synonyms[token]
    if (syns) for (const s of syns) expanded.add(s)
  }

  return [...expanded].filter((t) => t.length > 1)
}

function isPointerText(text: string): boolean {
  return /as\s+(?:defined|set\s+forth|specified|described|referenced|stated)\s+in|see\s+(?:section|article|exhibit|schedule|annex|appendix)|referenced\s+in\s+this\s+agreement|more\s+fully\s+set\s+forth|pursuant\s+to\s+section|shall\s+be\s+specified\s+and\s+agreed/i.test(
    text,
  )
}

function isSubstanceText(text: string): boolean {
  return /shall\s+(?:provide|perform|deliver|include|consist)|includes?\s+but\s+not\s+limited|deliverables?|services?\s+(?:include|consist)|scope\s+of\s+(?:work|services)|the\s+consultant\s+shall|responsibilities\s+(?:include|are)|consists?\s+of/i.test(
    text,
  )
}

function extractSectionRefs(text: string): string[] {
  const refs: string[] = []
  const re = /\b(?:section|article|exhibit|schedule|annex|appendix)\s+([0-9]+|[A-Z])\b/gi
  let m: RegExpExecArray | null
  while ((m = re.exec(text)) !== null) {
    refs.push(m[0].toLowerCase())
  }
  return [...new Set(refs)]
}

const CHART_HINT =
  /timeline|roadmap|gantt|chart|graph|figure|diagram|schedule|milestone|phase|quarter|q[1-4]\b/i

function scorePiece(piece: DocPiece, queryTokens: string[], question: string): number {
  if (!queryTokens.length) return 0
  const lower = piece.text.toLowerCase()
  const titleLower = (piece.title || '').toLowerCase()
  const qLower = question.toLowerCase()
  const tokens = new Set(tokenize(piece.text).flatMap((t) => [t, stem(t)]))
  let score = 0
  let covered = 0
  const visual =
    /\[(?:PAGE )?IMAGE|CHART \/ FIGURE|EMBEDDED IMAGE/i.test(piece.text) ||
    CHART_HINT.test(piece.text)
  const wantsVisual = /timeline|roadmap|gantt|chart|graph|figure|diagram|schedule/i.test(question)
  if (wantsVisual && visual) score += 18

  for (const token of queryTokens) {
    const stemmed = stem(token)
    const inBody = tokens.has(token) || tokens.has(stemmed) || lower.includes(token)
    const inTitle = titleLower.includes(token) || titleLower.includes(stemmed)
    if (inBody) {
      covered += 1
      score += 5
      const re = new RegExp(`\\b${token.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'gi')
      const hits = lower.match(re)
      if (hits) score += Math.min(hits.length, 8)
    }
    if (inTitle) score += 12
  }

  if (queryTokens.length > 0) {
    score += (covered / queryTokens.length) * 20
  }

  const words = qLower.replace(/[^a-z0-9\s']/g, ' ').split(/\s+/).filter((t) => t.length > 3)
  for (let i = 0; i < words.length - 1; i++) {
    const bigram = `${words[i]} ${words[i + 1]}`
    if (lower.includes(bigram) || titleLower.includes(bigram)) score += 10
  }

  // Prefer paragraphs that state the substance, not just "see Section X"
  if (isPointerText(piece.text) && !isSubstanceText(piece.text)) score -= 22
  if (isSubstanceText(piece.text)) score += 16
  if (/scope of (work|services)/i.test(piece.text) || /scope of (work|services)/i.test(titleLower)) {
    score += 12
  }

  const focus = question
    .replace(/^(what(?:'s| is| are)|define|explain|who is|tell me about)\s+/i, '')
    .replace(/[?]/g, '')
    .trim()
    .toLowerCase()
  if (focus.length >= 3 && (lower.includes(focus) || titleLower.includes(focus))) {
    score += 36
  }

  return score
}

function cleanSnippet(text: string): string {
  return text
    .replace(/\[\[\[PAGE[\s\S]*?\]\]\]/g, '')
    .replace(/\[CHART \/ FIGURE OCR\]/gi, '')
    .replace(/\[(?:PAGE )?IMAGE[^\]]*\]/gi, '')
    .replace(/\[EMBEDDED IMAGE[^\]]*\]/gi, '')
    .replace(/\[DOCX IMAGE[^\]]*\]/gi, '')
    .replace(/\s+/g, ' ')
    .trim()
}

function splitParagraphs(text: string): string[] {
  const cleaned = text
    .replace(/\[\[\[PAGE[\s\S]*?\]\]\]\s*/g, '')
    .replace(/\r\n/g, '\n')
  const parts = cleaned
    .split(/\n{2,}|(?<=\.)\s+(?=[A-Z])/)
    .map((p) => p.replace(/\s+/g, ' ').trim())
    .filter((p) => p.length >= 40)
  if (parts.length) return parts
  const fallback = cleaned.replace(/\s+/g, ' ').trim()
  return fallback ? [fallback.slice(0, 500)] : []
}

function pickSnippet(pageText: string, queryTokens: string[], question: string): string {
  const paras = splitParagraphs(pageText)
  if (!paras.length) return cleanSnippet(pageText).slice(0, 220)
  const ranked = paras
    .map((p) => ({
      p,
      score: scorePiece({ file: '', text: p }, queryTokens, question),
    }))
    .sort((a, b) => b.score - a.score)
  // Prefer a substance paragraph over a cross-reference pointer
  const best =
    ranked.find((r) => isSubstanceText(r.p) || !isPointerText(r.p))?.p ||
    ranked[0]?.p ||
    paras[0]
  const snippet = cleanSnippet(best)
  if (snippet.length <= 280) return snippet
  return `${snippet.slice(0, 270).trim()}…`
}

function selectRelevantContext(
  files: AnalyzedFile[],
  question: string,
  provider: ProviderId,
): {
  corpus: string
  hitCount: number
  citations: string[]
  sourceRefs: { page: number; title: string; label: string; snippet?: string }[]
} {
  const budget = ASK_CONTEXT_CHARS[provider]
  const queryTokens = expandQueryTokens(question, extraTokensFromDocument(question, files))
  const isOverview = /summar|overview|key (points|facts|takeaways)|main (points|ideas)|tl;?dr|what is (this|the) (document|file)|explain (this|the) (document|file)/i.test(
    question,
  )
  const isVisual = /timeline|roadmap|gantt|chart|graph|figure|diagram|schedule/i.test(question)

  type RankedPara = {
    file: string
    page: number
    title: string
    text: string
    score: number
  }

  const paraRows: RankedPara[] = []
  for (const file of files) {
    for (const page of file.pages || []) {
      const paras = splitParagraphs(page.text)
      for (const para of paras) {
        paraRows.push({
          file: file.name,
          page: page.page,
          title: page.title,
          text: para,
          score: scorePiece(
            { file: file.name, page: page.page, title: page.title, text: para },
            queryTokens,
            question,
          ),
        })
      }
      // Also score the whole page lightly so short-answer pages still surface
      paraRows.push({
        file: file.name,
        page: page.page,
        title: page.title,
        text: page.text.slice(0, 900),
        score:
          scorePiece(
            { file: file.name, page: page.page, title: page.title, text: page.text },
            queryTokens,
            question,
          ) * 0.65,
      })
    }
  }
  paraRows.sort((a, b) => b.score - a.score || a.page - b.page)

  const chosen: RankedPara[] = []
  let used = 0
  const maxParas = provider === 'browser' ? (isOverview || isVisual ? 8 : 8) : 14
  const perParaCap = provider === 'browser' ? 1_200 : 1_800

  const addPara = (row: RankedPara) => {
    const body = row.text.slice(0, perParaCap)
    if (
      chosen.some(
        (c) => c.page === row.page && c.file === row.file && c.text.slice(0, 80) === body.slice(0, 80),
      )
    ) {
      return false
    }
    const block = `----- PAGE ${row.page}${row.title ? ` · ${row.title}` : ''} -----\n${body}`
    if (used + block.length > budget) return false
    chosen.push({ ...row, text: body })
    used += block.length + 2
    return true
  }

  if (isOverview) {
    for (const file of files) {
      for (const page of (file.pages || []).slice(0, 4)) {
        addPara({
          file: file.name,
          page: page.page,
          title: page.title,
          text: page.text.slice(0, perParaCap),
          score: 1,
        })
      }
    }
  }

  if (isVisual) {
    for (const file of files) {
      for (const page of file.pages || []) {
        if (!/CHART \/ FIGURE|PAGE IMAGE OCR|EMBEDDED IMAGE/i.test(page.text) && !CHART_HINT.test(page.text)) {
          continue
        }
        addPara({
          file: file.name,
          page: page.page,
          title: page.title,
          text: page.text.slice(0, perParaCap),
          score: 25,
        })
      }
    }
  }

  const hasPositive = paraRows.some((r) => r.score > 0)
  for (const row of paraRows) {
    if (hasPositive && row.score <= 0 && chosen.length >= 3) break
    // Prefer substance over cross-reference-only blurbs when filling the budget
    if (
      chosen.length >= 2 &&
      isPointerText(row.text) &&
      !isSubstanceText(row.text) &&
      paraRows.some((r) => r.score > 0 && isSubstanceText(r.text))
    ) {
      continue
    }
    if (!addPara(row)) {
      if (used > budget * 0.9) break
      continue
    }
    if (chosen.length >= maxParas) break
  }

  // If hits point to "Section X", pull paragraphs that look like that section's content
  const sectionRefs = [
    ...new Set(chosen.flatMap((c) => extractSectionRefs(c.text))),
  ]
  if (sectionRefs.length && chosen.length < maxParas) {
    for (const file of files) {
      for (const page of file.pages || []) {
        const pageLower = `${page.title}\n${page.text}`.toLowerCase()
        const matchesSection = sectionRefs.some((ref) => pageLower.includes(ref))
        if (!matchesSection) continue
        for (const para of splitParagraphs(page.text)) {
          if (isPointerText(para) && !isSubstanceText(para)) continue
          addPara({
            file: file.name,
            page: page.page,
            title: page.title,
            text: para,
            score: 40,
          })
          if (chosen.length >= maxParas) break
        }
        if (chosen.length >= maxParas) break
      }
      if (chosen.length >= maxParas) break
    }
  }

  // Ensure at least one substance paragraph is included when available
  if (!chosen.some((c) => isSubstanceText(c.text))) {
    const substance = paraRows.find((r) => r.score > 0 && isSubstanceText(r.text))
    if (substance) addPara(substance)
  }

  if (chosen.length === 0) {
    for (const file of files) {
      for (const page of (file.pages || []).slice(0, 2)) {
        addPara({
          file: file.name,
          page: page.page,
          title: page.title,
          text: page.text.slice(0, perParaCap),
          score: 0,
        })
      }
    }
  }

  // Build one source card per page with the best matching statement
  const byPage = new Map<number, RankedPara>()
  for (const row of chosen) {
    const prev = byPage.get(row.page)
    if (!prev || row.score > prev.score) byPage.set(row.page, row)
  }

  const sourceRefs = [...byPage.values()]
    .sort((a, b) => a.page - b.page)
    .map((p) => {
      const fullPage =
        files.flatMap((f) => f.pages || []).find((pg) => pg.page === p.page)?.text || p.text
      const snippet = pickSnippet(fullPage, queryTokens, question) || cleanSnippet(p.text).slice(0, 220)
      const title = p.title || `Page ${p.page}`
      return {
        page: p.page,
        title,
        label: `Page ${p.page} — ${title}`,
        snippet,
      }
    })

  return {
    corpus: chosen
      .map((p) => `----- PAGE ${p.page}${p.title ? ` · ${p.title}` : ''} -----\n${p.text}`)
      .join('\n\n'),
    hitCount: chosen.length,
    citations: sourceRefs.map((s) => s.label),
    sourceRefs,
  }
}

async function callOpenAiCompatible(
  baseUrl: string,
  apiKey: string,
  model: string,
  prompt: string,
  extraHeaders: Record<string, string> = {},
): Promise<string> {
  const res = await fetch(`${baseUrl}/chat/completions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
      ...extraHeaders,
    },
    body: JSON.stringify({
      model,
      temperature: 0.15,
      messages: [
        {
          role: 'system',
          content:
            'You are ScanAsk, a document analyst. Use only the provided document content. Be precise and specific. Never give generic answers.',
        },
        { role: 'user', content: prompt },
      ],
    }),
  })

  const data = (await res.json()) as {
    error?: { message?: string }
    choices?: { message?: { content?: string } }[]
  }

  if (!res.ok) {
    throw new Error(data.error?.message || `API request failed (${res.status})`)
  }

  const text = data.choices?.[0]?.message?.content || ''
  if (!text.trim()) throw new Error('The model returned an empty response. Try again.')
  return text.trim()
}

async function callGemini(apiKey: string, model: string, prompt: string): Promise<string> {
  const models = [model, 'gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-2.5-flash']
  const tried = new Set<string>()
  let lastError: Error | null = null

  for (const candidate of models) {
    if (tried.has(candidate)) continue
    tried.add(candidate)

    const url = `https://generativelanguage.googleapis.com/v1beta/models/${candidate}:generateContent?key=${encodeURIComponent(apiKey)}`
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contents: [{ role: 'user', parts: [{ text: prompt }] }],
        generationConfig: {
          temperature: 0.15,
          maxOutputTokens: 8192,
        },
      }),
    })

    const data = (await res.json()) as {
      error?: { message?: string }
      candidates?: { content?: { parts?: { text?: string }[] } }[]
    }

    if (!res.ok) {
      lastError = new Error(data.error?.message || `Gemini request failed (${res.status})`)
      // Try next model on not-found / unsupported
      if (/not found|not supported|invalid/i.test(lastError.message)) continue
      throw lastError
    }

    const text = data.candidates?.[0]?.content?.parts?.map((p) => p.text || '').join('') || ''
    if (!text.trim()) {
      lastError = new Error('The model returned an empty response. Try again.')
      continue
    }
    return text.trim()
  }

  throw lastError || new Error('Gemini request failed.')
}

async function generateViaProxy(config: LlmConfig, prompt: string): Promise<string | null> {
  try {
    const res = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        provider: config.provider,
        apiKey: config.apiKey,
        model: config.model,
        prompt,
      }),
    })

    if (res.status === 404) return null

    const data = (await res.json()) as { text?: string; error?: string }
    if (!res.ok) throw new Error(data.error || `Proxy request failed (${res.status})`)
    if (!data.text?.trim()) throw new Error('The model returned an empty response. Try again.')
    return data.text.trim()
  } catch (err) {
    if (err instanceof TypeError) return null
    throw err
  }
}

async function generateDirect(config: LlmConfig, prompt: string): Promise<string> {
  const key = config.apiKey.trim()
  if (config.provider === 'gemini') {
    return callGemini(key, config.model || 'gemini-2.0-flash', prompt)
  }
  if (config.provider === 'openrouter') {
    return callOpenAiCompatible(
      'https://openrouter.ai/api/v1',
      key,
      config.model || OPENROUTER_DEFAULT_MODEL,
      prompt,
      {
        'HTTP-Referer': typeof window !== 'undefined' ? window.location.origin : 'http://localhost',
        'X-Title': 'ScanAsk',
      },
    )
  }
  if (config.provider === 'groq') {
    return callOpenAiCompatible(
      'https://api.groq.com/openai/v1',
      key,
      config.model || 'llama-3.1-8b-instant',
      prompt,
    )
  }
  return callOpenAiCompatible(
    'https://api.openai.com/v1',
    key,
    config.model || 'gpt-4o-mini',
    prompt,
  )
}

function isRateLimitError(err: unknown): boolean {
  const msg = err instanceof Error ? err.message : String(err)
  return /rate limit|tokens per minute|tpm|try again in/i.test(msg)
}

function isTooLargeError(err: unknown): boolean {
  const msg = err instanceof Error ? err.message : String(err)
  return /too large|request too large|context.?length|maximum context/i.test(msg)
}

function parseRetrySeconds(err: unknown): number {
  const msg = err instanceof Error ? err.message : String(err)
  const match = msg.match(/try again in\s+([\d.]+)\s*s/i)
  if (match) return Math.ceil(parseFloat(match[1]) + 1)
  return 16
}

async function generateOnce(config: LlmConfig, prompt: string): Promise<string> {
  if (config.provider === 'browser') {
    return browserChat(
      'Answer from the document passages only. Copy product names and acronyms EXACTLY (never invent near-miss names). Use chart/figure OCR as real content. Do not mention excerpts. Be specific.',
      prompt,
    )
  }

  const proxied = await generateViaProxy(config, prompt)
  if (proxied) return proxied
  return generateDirect(config, prompt)
}

async function generate(config: LlmConfig, prompt: string): Promise<string> {
  if (config.provider !== 'browser' && !config.apiKey?.trim()) {
    throw new Error(
      'Add an API key in Settings, or switch Provider to Browser LLM (no key needed).',
    )
  }

  let payload = prompt
  let lastError: unknown
  const maxPayload = config.provider === 'browser' ? 6_500 : 6_000

  for (let attempt = 0; attempt < 5; attempt++) {
    try {
      return await generateOnce(config, payload)
    } catch (err) {
      lastError = err

      if (isRateLimitError(err)) {
        const waitSec = parseRetrySeconds(err)
        await sleep(waitSec * 1000)
        continue
      }

      if (isTooLargeError(err) && payload.length > maxPayload) {
        payload =
          payload.slice(0, Math.floor(payload.length * 0.6)) +
          '\n\n[Document truncated to fit model limits.]'
        continue
      }

      throw err
    }
  }

  throw lastError instanceof Error
    ? lastError
    : new Error('Generation failed. Try again.')
}

/** Pull start + evenly spaced middle + end so one request still covers the whole file. */
function stratifiedSample(files: { name: string; text: string }[], budget: number): string {
  const parts: string[] = []
  let remaining = budget

  for (const file of files) {
    if (remaining <= 0) break
    const header = `===== FILE: ${file.name} =====\n`
    const bodyBudget = Math.max(remaining - header.length, 0)
    if (bodyBudget < 500) break

    const text = file.text.trim()
    if (text.length <= bodyBudget) {
      parts.push(header + text)
      remaining -= header.length + text.length
      continue
    }

    const window = Math.floor(bodyBudget / 4)
    const start = text.slice(0, window)
    const midAStart = Math.floor(text.length * 0.33) - Math.floor(window / 2)
    const midBStart = Math.floor(text.length * 0.66) - Math.floor(window / 2)
    const midA = text.slice(Math.max(0, midAStart), Math.max(0, midAStart) + window)
    const midB = text.slice(Math.max(0, midBStart), Math.max(0, midBStart) + window)
    const end = text.slice(-window)

    const sample = [
      start,
      '\n\n[...]\n\n',
      midA,
      '\n\n[...]\n\n',
      midB,
      '\n\n[...]\n\n',
      end,
    ].join('')

    parts.push(header + sample.slice(0, bodyBudget))
    remaining -= header.length + Math.min(sample.length, bodyBudget)
  }

  return parts.join('\n\n')
}

function extractJson(raw: string): unknown {
  const fenced = raw.match(/```(?:json)?\s*([\s\S]*?)```/)
  const candidate = fenced ? fenced[1] : raw
  const start = candidate.indexOf('{')
  const end = candidate.lastIndexOf('}')
  if (start === -1 || end === -1) throw new Error('Could not parse the training analysis. Try again.')
  return JSON.parse(candidate.slice(start, end + 1))
}

function normalizeBrief(raw: unknown): KnowledgeBrief {
  const obj = (raw && typeof raw === 'object' ? raw : {}) as Record<string, unknown>

  const entities = Array.isArray(obj.entities)
    ? obj.entities.map((e) => {
        if (typeof e === 'string') return { name: e }
        const row = e as Record<string, unknown>
        return {
          name: String(row.name || row.entity || 'Unknown'),
          role: row.role ? String(row.role) : undefined,
        }
      })
    : []

  const sections = Array.isArray(obj.sections)
    ? obj.sections.map((s) => {
        const row = (s && typeof s === 'object' ? s : {}) as Record<string, unknown>
        return {
          title: String(row.title || 'Section'),
          gist: String(row.gist || row.summary || ''),
        }
      })
    : []

  return {
    summary: String(obj.summary || ''),
    documentType: String(obj.documentType || obj.type || 'document'),
    topics: Array.isArray(obj.topics) ? obj.topics.map(String) : [],
    entities,
    keyFacts: Array.isArray(obj.keyFacts) ? obj.keyFacts.map(String) : [],
    dates: Array.isArray(obj.dates) ? obj.dates.map(String) : [],
    sections,
  }
}

function mergeBriefs(briefs: KnowledgeBrief[]): KnowledgeBrief {
  const uniq = (items: string[]) => [...new Set(items.map((s) => s.trim()).filter(Boolean))]

  const entityMap = new Map<string, string | undefined>()
  for (const b of briefs) {
    for (const e of b.entities) {
      if (!entityMap.has(e.name)) entityMap.set(e.name, e.role)
    }
  }

  const sections = briefs.flatMap((b) => b.sections).slice(0, 24)
  const summaries = briefs.map((b) => b.summary).filter(Boolean)

  return {
    summary: summaries.slice(0, 4).join(' '),
    documentType: briefs.find((b) => b.documentType)?.documentType || 'document',
    topics: uniq(briefs.flatMap((b) => b.topics)).slice(0, 16),
    entities: [...entityMap.entries()].slice(0, 30).map(([name, role]) => ({ name, role })),
    keyFacts: uniq(briefs.flatMap((b) => b.keyFacts)).slice(0, 40),
    dates: uniq(briefs.flatMap((b) => b.dates)).slice(0, 24),
    sections,
  }
}

const BRIEF_SHAPE = `{
  "summary": "2-4 sentence overview of this part",
  "documentType": "contract | report | resume | policy | article | other",
  "topics": ["..."],
  "entities": [{"name":"...","role":"..."}],
  "keyFacts": ["specific fact 1", "specific fact 2"],
  "dates": ["..."],
  "sections": [{"title":"...","gist":"what this section says"}]
}`

async function analyzeChunk(
  config: LlmConfig,
  piece: DocPiece,
  index: number,
  total: number,
): Promise<KnowledgeBrief> {
  const prompt = `You are ScanAsk. Analyze PART ${index + 1} of ${total} from file "${piece.file}".
Extract every useful specific detail from this part only.

Return ONLY valid JSON with this shape:
${BRIEF_SHAPE}

DOCUMENT PART:
${piece.text}`

  const raw = await generate(config, prompt)
  const brief = normalizeBrief(extractJson(raw))
  if (!brief.summary) brief.summary = raw.slice(0, 400)
  return brief
}

export function resolveLlmConfig(
  userKey?: string,
  userProvider?: ProviderId,
): LlmConfig | null {
  const openrouterEnv =
    (import.meta.env.VITE_OPENROUTER_API_KEY as string | undefined)?.trim() || ''
  const groqEnv = (import.meta.env.VITE_GROQ_API_KEY as string | undefined)?.trim() || ''
  const geminiEnv = (import.meta.env.VITE_GEMINI_API_KEY as string | undefined)?.trim() || ''
  const openaiEnv = (import.meta.env.VITE_OPENAI_API_KEY as string | undefined)?.trim() || ''

  const key = userKey?.trim() || ''
  const provider = userProvider || 'browser'

  if (provider === 'browser') {
    return { provider: 'browser', apiKey: '', model: BROWSER_MODEL_ID }
  }

  if (key) {
    const defaults: Record<Exclude<ProviderId, 'browser'>, string> = {
      openrouter: OPENROUTER_DEFAULT_MODEL,
      groq: 'llama-3.1-8b-instant',
      gemini: 'gemini-2.0-flash',
      openai: 'gpt-4o-mini',
    }
    return { provider, apiKey: key, model: defaults[provider] }
  }

  if (openrouterEnv) {
    return {
      provider: 'openrouter',
      apiKey: openrouterEnv,
      model:
        (import.meta.env.VITE_OPENROUTER_MODEL as string | undefined) ||
        OPENROUTER_DEFAULT_MODEL,
    }
  }
  if (geminiEnv) {
    return {
      provider: 'gemini',
      apiKey: geminiEnv,
      model: (import.meta.env.VITE_GEMINI_MODEL as string | undefined) || 'gemini-2.0-flash',
    }
  }
  if (groqEnv) {
    return {
      provider: 'groq',
      apiKey: groqEnv,
      model: (import.meta.env.VITE_GROQ_MODEL as string | undefined) || 'llama-3.1-8b-instant',
    }
  }
  if (openaiEnv) {
    return {
      provider: 'openai',
      apiKey: openaiEnv,
      model: (import.meta.env.VITE_OPENAI_MODEL as string | undefined) || 'gpt-4o-mini',
    }
  }

  // Always fall back to local browser LLM — no key required
  return { provider: 'browser', apiKey: '', model: BROWSER_MODEL_ID }
}

function uniqueStrings(items: string[], cap: number): string[] {
  const seen = new Set<string>()
  const out: string[] = []
  for (const raw of items) {
    const s = raw.replace(/\s+/g, ' ').trim()
    if (!s || seen.has(s.toLowerCase())) continue
    seen.add(s.toLowerCase())
    out.push(s)
    if (out.length >= cap) break
  }
  return out
}

function buildExtractiveBrief(files: AnalyzedFile[]): KnowledgeBrief {
  const pages = files.flatMap((f) => f.pages || [])
  const full = files.map((f) => f.text).join('\n')
  const lead = pages
    .slice(0, 4)
    .map((p) =>
      p.text
        .replace(/\[(?:PAGE )?IMAGE[^\]]*\]/gi, '')
        .replace(/\[CHART \/ FIGURE OCR\]/gi, '')
        .replace(/\s+/g, ' ')
        .trim(),
    )
    .filter(Boolean)
    .join(' ')

  const dates = uniqueStrings(
    [...full.matchAll(/\b(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}|Q[1-4]\s*20\d{2}|FY\s*20\d{2}|\d{4}-\d{2}-\d{2}|\d{1,2}\/\d{1,2}\/\d{2,4})\b/gi)].map(
      (m) => m[0],
    ),
    16,
  )

  const entities: { name: string; role?: string }[] = []
  for (const [acro, expansion] of collectAcronyms(files)) {
    entities.push({ name: expansion, role: acro.toUpperCase() })
  }
  const nameHits = [
    ...full.matchAll(/\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z0-9]+){1,4})\b/g),
  ]
    .map((m) => m[1])
    .filter((n) => n.length >= 8 && !/^Page\s/i.test(n))
  for (const name of uniqueStrings(nameHits, 18)) {
    if (!entities.some((e) => e.name.toLowerCase() === name.toLowerCase())) {
      entities.push({ name })
    }
  }

  const factLines = pages
    .slice(0, 12)
    .flatMap((p) =>
      p.text
        .split(/(?<=[.!?])\s+/)
        .map((s) => s.replace(/\s+/g, ' ').trim())
        .filter((s) => s.length > 50 && s.length < 280 && /\d|[A-Z]{2,}/.test(s)),
    )

  const sections = pages.slice(0, 16).map((p) => ({
    title: p.title || `Page ${p.page}`,
    gist: p.text.replace(/\s+/g, ' ').trim().slice(0, 180),
    page: p.page,
  }))

  const topics = uniqueStrings(
    tokenize(lead)
      .filter((t) => t.length > 4)
      .slice(0, 40),
    12,
  )

  return {
    summary: (lead.slice(0, 520) || 'Document loaded for question answering.').trim(),
    documentType: /agreement|contract|sow/i.test(full)
      ? 'contract'
      : /timeline|roadmap/i.test(full)
        ? 'plan'
        : 'document',
    topics,
    entities: entities.slice(0, 24),
    keyFacts: uniqueStrings(factLines, 12),
    dates,
    sections,
  }
}

function lockDocumentNames(answer: string, document: string): string {
  const phrases = uniqueStrings(
    [...document.matchAll(/\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z0-9]+){2,4})\b/g)].map((m) => m[1]),
    40,
  ).sort((a, b) => b.length - a.length)

  let out = answer
  for (const phrase of phrases) {
    const words = phrase.split(/\s+/)
    if (words.length < 3) continue
    const prefix = words
      .slice(0, -1)
      .map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
      .join('\\s+')
    const re = new RegExp(`\\b${prefix}\\s+[A-Za-z][A-Za-z0-9+&-]*\\b`, 'g')
    out = out.replace(re, (match) => {
      if (match.toLowerCase() === phrase.toLowerCase()) return match
      if (document.toLowerCase().includes(match.toLowerCase())) return match
      return phrase
    })
  }
  return out
}

function isModelRefusal(answer: string): boolean {
  return /couldn'?t find|do not provide|does not provide|doesn't provide|no information in the (excerpt|passage|document)|excerpts? (do not|don't|did not)|not (mentioned|stated) in the (excerpt|passage)/i.test(
    answer,
  )
}

function extractiveFallback(
  _question: string,
  retrieved: { corpus: string; sourceRefs: { page: number; title: string; snippet?: string }[] },
): string {
  const bits = retrieved.sourceRefs
    .map((s) => s.snippet)
    .filter((s): s is string => Boolean(s && s.length > 40))
    .slice(0, 5)
  if (!bits.length) {
    const plain = retrieved.corpus.replace(/----- PAGE[\s\S]*?-----\n/g, '').replace(/\s+/g, ' ').trim()
    if (plain.length > 80) {
      return `From the document:\n\n${plain.slice(0, 900)}${plain.length > 900 ? '…' : ''}`
    }
    return "I couldn't find that in your uploaded file."
  }
  return `Here is what the document says about this:\n\n${bits.map((b) => `• ${b}`).join('\n\n')}`
}

const REAL_DATE =
  /\b(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s*\d{2,4}|Q[1-4]\s*'?-?\s*(?:20)?\d{2}|FY\s*(?:20)?\d{2}|20\d{2}\s*[–-]\s*20\d{2}|week\s+\d+)\b/i

function collapseRepeatedText(text: string): string {
  let t = text.replace(/\s+/g, ' ').trim()
  for (let n = 2; n <= 4; n++) {
    const size = Math.floor(t.length / n)
    if (size < 36) continue
    const chunk = t.slice(0, size).trim()
    if (chunk.length >= 36 && t === Array.from({ length: n }, () => chunk).join(' ')) {
      return chunk
    }
  }
  return t
}

function nearDuplicate(a: string, b: string): boolean {
  const x = a.toLowerCase().replace(/\s+/g, ' ')
  const y = b.toLowerCase().replace(/\s+/g, ' ')
  if (x === y) return true
  if (x.length > 24 && y.length > 24 && (x.includes(y) || y.includes(x))) return true
  const aw = x.split(' ').slice(0, 10).join(' ')
  const bw = y.split(' ').slice(0, 10).join(' ')
  return aw.length > 20 && aw === bw
}

function splitDocSentences(text: string): string[] {
  const cleaned = collapseRepeatedText(cleanSnippet(text))
  const parts = cleaned
    .split(/\n+|[\u2022•●]\s*|(?<=[.!?])\s+(?=[A-Z0-9])|(?=\d+\.\s+)/)
    .map((s) => collapseRepeatedText(s.replace(/^[\d.)\-\s]+/, ' ').trim()))
    .filter((s) => s.length >= 18 && s.length <= 320)
  const unique: string[] = []
  for (const part of parts) {
    if (unique.some((u) => nearDuplicate(u, part))) continue
    unique.push(part)
  }
  return unique.length ? unique : cleaned ? [cleaned.slice(0, 220)] : []
}

function clip(text: string, max = 180): string {
  const t = collapseRepeatedText(text)
  if (t.length <= max) return t
  return `${t.slice(0, max).replace(/\s+\S*$/, '').trim()}…`
}

function formatStructuredAnswer(title: string, lead: string, items: string[]): string {
  const lines = [`**${title}**`]
  if (lead) lines.push(lead, '')
  if (items.length) {
    lines.push('**From the file**')
    items.forEach((item, i) => lines.push(`${i + 1}. ${item}`))
  }
  return lines.join('\n')
}

function questionFocus(question: string): string {
  return question
    .replace(/^(what(?:'s| is| are)|define|explain|who is|tell me about|list|show|summarize)\s+/i, '')
    .replace(/[?]/g, '')
    .trim()
}

function buildGroundedAnswer(
  question: string,
  files: AnalyzedFile[],
  brain: DocumentBrain | null,
): {
  answer: string
  sources: string[]
  sourceRefs: { page: number; title: string; label: string; snippet?: string }[]
} {
  const extra = extraTokensFromDocument(question, files)
  const queryTokens = expandQueryTokens(question, extra)
  const focus = questionFocus(question)
  const isOverview = /summar|overview|key (points|facts|takeaways)|main (points|ideas)|tl;?dr|what is (this|the) (document|file)/i.test(
    question,
  )
  const isDates = /date|timeline|roadmap|schedule|when|milestone|gantt|chart|graph|deadline/i.test(
    question,
  )

  type Hit = { page: number; title: string; text: string; score: number }
  const hits: Hit[] = []

  for (const file of files) {
    for (const page of file.pages || []) {
      for (const sentence of splitDocSentences(page.text)) {
        let score = scorePiece(
          { file: file.name, page: page.page, title: page.title, text: sentence },
          queryTokens,
          question,
        )
        const lower = sentence.toLowerCase()
        if (focus.length >= 2 && lower.includes(focus.toLowerCase())) score += 40
        for (const token of extra) {
          if (token.length > 1 && lower.includes(token.toLowerCase())) score += 12
        }
        if (/\b(is|are|means|refers to)\b/i.test(sentence) && focus.length >= 2 && lower.includes(focus.toLowerCase())) {
          score += 28
        }
        if (isDates && REAL_DATE.test(sentence)) score += 22
        if (/\bcognite data fusion\b/i.test(sentence) && /\bcdf\b/i.test(question)) score += 80
        if (/\bcognite data fusion\s*\(\s*cdf\s*\)/i.test(sentence)) score += 90
        if (/\bcdf\s*\(\s*cognite data fusion\s*\)/i.test(sentence)) score += 90
        if (score > 0) {
          hits.push({
            page: page.page,
            title: page.title,
            text: sentence,
            score,
          })
        }
      }
    }
  }

  hits.sort((a, b) => b.score - a.score || a.page - b.page)

  const picked: Hit[] = []
  const seen = new Set<string>()
  const perPage = new Map<number, number>()
  for (const hit of hits) {
    const key = hit.text.slice(0, 72).toLowerCase()
    if (seen.has(key)) continue
    const n = perPage.get(hit.page) || 0
    if (n >= (isDates ? 4 : 2)) continue
    seen.add(key)
    perPage.set(hit.page, n + 1)
    picked.push(hit)
    if (picked.length >= (isDates ? 10 : 7)) break
  }

  const toRefs = (rows: Hit[]) => {
    const byPage = new Map<number, Hit>()
    for (const row of rows) {
      const prev = byPage.get(row.page)
      if (!prev || row.score > prev.score) byPage.set(row.page, row)
    }
    return [...byPage.values()]
      .sort((a, b) => a.page - b.page)
      .slice(0, 4)
      .map((p) => ({
        page: p.page,
        title: p.title || `Page ${p.page}`,
        label: `Page ${p.page} — ${p.title || `Page ${p.page}`}`,
        snippet: clip(p.text, 120),
      }))
  }

  if (isOverview) {
    const items: string[] = []
    if (brain?.brief.keyFacts.length) {
      for (const fact of brain.brief.keyFacts.slice(0, 6)) items.push(clip(fact, 160))
    }
    const leadPages = files.flatMap((f) => (f.pages || []).slice(0, 3))
    if (!items.length) {
      for (const page of leadPages) {
        const snippet = clip(cleanSnippet(page.text), 160)
        if (snippet) items.push(snippet)
      }
    }
    const lead =
      brain?.brief.summary
        ? clip(brain.brief.summary, 280)
        : items[0] || 'This file was loaded for question answering.'
    const names = (brain?.brief.entities || [])
      .slice(0, 8)
      .map((e) => (e.role ? `${e.name} (${e.role})` : e.name))
    const extra = names.length ? [`Names used: ${names.join('; ')}`] : []
    const dates = (brain?.brief.dates || []).slice(0, 8)
    if (dates.length) extra.push(`Dates named: ${dates.join('; ')}`)
    const refs = toRefs(
      leadPages.map((p) => ({
        page: p.page,
        title: p.title,
        text: clip(cleanSnippet(p.text), 120),
        score: 1,
      })),
    )
    return {
      answer: formatStructuredAnswer('Answer', lead, [...items.slice(0, 6), ...extra]),
      sources: refs.map((r) => r.label),
      sourceRefs: refs,
    }
  }

  if (isDates) {
    const dated = hits.filter((h) => REAL_DATE.test(h.text))
    const scope = picked.filter((h) => !dated.some((d) => d.text === h.text))
    const dateItems = (dated.length ? dated : picked)
      .map((h) => clip(h.text, 160))
      .filter((t, i, arr) => arr.findIndex((x) => nearDuplicate(x, t)) === i)
      .slice(0, 8)
    const lead =
      dated.length > 0
        ? 'The file names these dates and timeline items:'
        : 'The file describes timeline and scope work as follows:'
    const refs = toRefs(dated.length ? dated : picked)
    return {
      answer: formatStructuredAnswer('Timeline', lead, dateItems.length ? dateItems : scope.map((h) => clip(h.text, 160)).slice(0, 6)),
      sources: refs.map((r) => r.label),
      sourceRefs: refs,
    }
  }

  if (!picked.length) {
    return {
      answer: "I couldn't find that in your uploaded file.",
      sources: [],
      sourceRefs: [],
    }
  }

  const glossary = collectAcronyms(files)
  const glossaryLead: string[] = []
  for (const [acro, expansion] of glossary) {
    if (new RegExp(`\\b${acro.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'i').test(question)) {
      glossaryLead.push(`In this file, ${acro.toUpperCase()} is ${expansion}.`)
    }
  }

  const details = picked
    .map((h) => clip(h.text, 170))
    .filter((t, i, arr) => arr.findIndex((x) => nearDuplicate(x, t)) === i)
    .slice(0, 6)

  const leadText = glossaryLead[0] || details[0] || ''
  const rest = glossaryLead[0] ? details : details.slice(1)
  const refs = toRefs(picked)
  return {
    answer: formatStructuredAnswer('Answer', leadText, rest),
    sources: refs.map((r) => r.label),
    sourceRefs: refs,
  }
}

export async function trainOnDocuments(
  files: AnalyzedFile[],
  config: LlmConfig,
): Promise<DocumentBrain> {
  const ready = files.filter((f) => f.text.trim())
  if (!ready.length) throw new Error('No document text to train on.')

  // Browser LLM: extractive brief from the real text (fast, keeps names like CDF exact).
  if (config.provider === 'browser') {
    return {
      brief: buildExtractiveBrief(ready),
      fileNames: ready.map((f) => f.name),
      trainedAt: Date.now(),
    }
  }

  // OpenRouter / Gemini: send a large slice of the full document in one training pass.
  if (config.provider === 'openrouter' || config.provider === 'gemini') {
    const corpus = stratifiedSample(
      ready.map((f) => ({ name: f.name, text: f.text })),
      LARGE_TRAIN_CHARS,
    )

    const fullJoin = ready
      .map((f) => `===== FILE: ${f.name} =====\n${f.text}`)
      .join('\n\n')
    const documentBlock =
      fullJoin.length <= LARGE_TRAIN_CHARS ? fullJoin : corpus

    const prompt = `You are ScanAsk. Thoroughly read and learn EVERYTHING in the document(s) below.
Build a durable knowledge brief for answering user questions later.

Rules:
- Use only information present in the documents.
- Be specific: names, numbers, dates, clauses, definitions, obligations, findings.
- Cover the whole document, not just the beginning.
- Return ONLY valid JSON with this shape:
${BRIEF_SHAPE}

DOCUMENTS:
${documentBlock}`

    const raw = await generate(config, prompt)
    const brief = normalizeBrief(extractJson(raw))
    if (!brief.summary) brief.summary = raw.slice(0, 600)

    return {
      brief,
      fileNames: ready.map((f) => f.name),
      trainedAt: Date.now(),
    }
  }

  // Groq free tier: ONE compact training call (avoids TPM pile-up).
  if (config.provider === 'groq') {
    const sample = stratifiedSample(
      ready.map((f) => ({ name: f.name, text: f.text })),
      GROQ_TRAIN_SAMPLE_CHARS,
    )

    const prompt = `You are ScanAsk. Thoroughly analyze the document sample below (start, middle, and end sections).
Build a knowledge brief for answering questions later.

Rules:
- Use only information present in the text.
- Be specific: names, numbers, dates, clauses, findings.
- Return ONLY valid JSON with this shape:
${BRIEF_SHAPE}

DOCUMENT SAMPLE:
${sample}`

    const raw = await generate(config, prompt)
    const brief = normalizeBrief(extractJson(raw))
    if (!brief.summary) brief.summary = raw.slice(0, 600)

    return {
      brief,
      fileNames: ready.map((f) => f.name),
      trainedAt: Date.now(),
    }
  }

  const pieces = buildPiecesFromFiles(ready, CHUNK_CHARS[config.provider])

  const maxPieces = 12
  const selected =
    pieces.length <= maxPieces
      ? pieces
      : [...pieces.slice(0, Math.ceil(maxPieces / 2)), ...pieces.slice(-Math.floor(maxPieces / 2))]

  const partials: KnowledgeBrief[] = []
  for (let i = 0; i < selected.length; i++) {
    partials.push(await analyzeChunk(config, selected[i], i, selected.length))
  }

  let brief = mergeBriefs(partials)

  if (partials.length > 1) {
    const mergePrompt = `You are ScanAsk. Merge these partial analyses into one knowledge brief.
Keep only facts present in the partials. Return ONLY valid JSON:
${BRIEF_SHAPE}

PARTIAL ANALYSES:
${JSON.stringify(partials, null, 2).slice(0, 20_000)}`

    try {
      const raw = await generate(config, mergePrompt)
      brief = normalizeBrief(extractJson(raw))
      if (!brief.summary) brief = mergeBriefs(partials)
    } catch {
      brief = mergeBriefs(partials)
    }
  }

  return {
    brief,
    fileNames: ready.map((f) => f.name),
    trainedAt: Date.now(),
  }
}

export async function buildLocalIndex(
  files: AnalyzedFile[],
  config: LlmConfig,
): Promise<DocumentBrain> {
  return trainOnDocuments(files, config)
}

export async function askWithBrain(
  question: string,
  files: AnalyzedFile[],
  brain: DocumentBrain | null,
  config: LlmConfig,
  history: { role: 'user' | 'assistant'; content: string }[] = [],
): Promise<{
  answer: string
  sources: string[]
  sourceRefs: { page: number; title: string; label: string; snippet?: string }[]
}> {
  const ready = files.filter((f) => f.text.trim())
  if (!ready.length) {
    return { answer: 'Upload a file first.', sources: [], sourceRefs: [] }
  }

  // Customer path: quote the file. Never wait on a local LLM that can invent names.
  if (config.provider === 'browser') {
    return buildGroundedAnswer(question, ready, brain)
  }

  const fullJoin = ready
    .map((f) => `===== FILE: ${f.name} =====\n${f.text}`)
    .join('\n\n')

  let retrieved: {
    corpus: string
    hitCount: number
    citations: string[]
    sourceRefs: { page: number; title: string; label: string; snippet?: string }[]
  }

  if (
    (config.provider === 'gemini' || config.provider === 'openrouter') &&
    fullJoin.length <= ASK_CONTEXT_CHARS[config.provider]
  ) {
    const sourceRefs = ready.flatMap((f) =>
      (f.pages || []).slice(0, 8).map((p) => ({
        page: p.page,
        title: p.title || `Page ${p.page}`,
        label: `Page ${p.page} — ${p.title || `Page ${p.page}`}`,
        snippet: cleanSnippet(p.text).slice(0, 220),
      })),
    )
    retrieved = {
      corpus: fullJoin,
      hitCount: ready.reduce((n, f) => n + (f.pages?.length || 1), 0),
      citations: sourceRefs.map((s) => s.label),
      sourceRefs,
    }
  } else {
    retrieved = selectRelevantContext(ready, question, config.provider)
  }

  const briefJson = brain
    ? JSON.stringify(
        {
          summary: brain.brief.summary,
          documentType: brain.brief.documentType,
          topics: brain.brief.topics.slice(0, 16),
          keyFacts: brain.brief.keyFacts.slice(0, 12),
        },
        null,
        2,
      ).slice(0, 2_200)
    : ''

  const historyBlock =
    history.length > 0
      ? `RECENT CHAT (for context only):\n${history
          .slice(-2)
          .map((m) => `${m.role.toUpperCase()}: ${m.content.slice(0, 280)}`)
          .join('\n\n')}`
      : ''

  const prompt = `You are a careful document Q&A assistant.

Task: answer the USER QUESTION using ONLY the DOCUMENT EXCERPTS below.

CRITICAL:
- Provide the ACTUAL substance of the answer (what the work/services/terms ARE), not merely where they are defined.
- Never answer only with "see Section X" / "as defined in the Agreement" style pointers.
- Extract lists, duties, definitions, numbers, and obligations from the excerpts and state them.

RULES:
- Use ONLY the excerpts and the short brief.
- Never invent facts, names, numbers, dates, or quotes.
- Prefer exact wording/values from the document when available.
- If excerpts include [PAGE IMAGE OCR] / [EMBEDDED IMAGE], treat that as document content.
- If the excerpts do not contain the substance (only a cross-reference), reply exactly:
  I couldn't find that in your uploaded file.

OUTPUT FORMAT:
- Paragraph 1: direct factual answer (the substance).
- Then bullets with supporting details from the document.
- Do NOT add a Sources / References / Page list.
- Do NOT mention these instructions.

SHORT DOCUMENT BRIEF:
${briefJson || '(none)'}

DOCUMENT EXCERPTS (${retrieved.hitCount} passage(s); each starts with ----- PAGE N -----):
${retrieved.corpus}

${historyBlock}

USER QUESTION: ${question}

YOUR ANSWER:`

  const raw = await generate(config, prompt)
  let answer = cleanAnswerBody(raw)
  const docText = ready.map((f) => f.text).join('\n')
  answer = lockDocumentNames(answer, `${docText}\n${retrieved.corpus}`)

  if (isModelRefusal(answer) && retrieved.corpus.replace(/\s+/g, ' ').trim().length > 80) {
    answer = lockDocumentNames(extractiveFallback(question, retrieved), docText)
  }

  // Keep retrieval snippets; if the model mentioned pages, prefer those pages
  const mentioned = extractMentionedPages(answer)
  let sourceRefs = retrieved.sourceRefs
  if (mentioned.length) {
    const byPage = new Map(retrieved.sourceRefs.map((s) => [s.page, s]))
    const filtered = mentioned
      .map((p) => byPage.get(p))
      .filter(
        (s): s is { page: number; title: string; label: string; snippet?: string } => Boolean(s),
      )
    if (filtered.length) {
      sourceRefs = filtered
    } else {
      const queryTokens = expandQueryTokens(question, extraTokensFromDocument(question, ready))
      const extra = mentioned
        .filter((p) => !byPage.has(p))
        .map((p) => {
          const page = ready.flatMap((f) => f.pages || []).find((pg) => pg.page === p)
          const title = page?.title || `Page ${p}`
          const snippet = page
            ? pickSnippet(page.text, queryTokens, question)
            : undefined
          return { page: p, title, label: `Page ${p} — ${title}`, snippet }
        })
      sourceRefs = [...retrieved.sourceRefs, ...extra].sort((a, b) => a.page - b.page)
    }
  }

  // Cap to the most relevant 4 reference cards
  sourceRefs = sourceRefs.slice(0, 4)

  // If the model only pointed to a section, replace with substance from retrieved snippets
  const pointerOnly =
    /defined in section|set forth in|specified and agreed|referenced in this agreement|see section/i.test(
      answer,
    ) && !isSubstanceText(answer)
  if (pointerOnly) {
    const substanceBits = [
      ...sourceRefs.map((s) => s.snippet || ''),
      ...retrieved.corpus.split(/----- PAGE/).map((p) => p.trim()),
    ]
      .map((t) => cleanSnippet(t))
      .filter((t) => t.length > 60 && (isSubstanceText(t) || !isPointerText(t)))
      .slice(0, 4)

    if (substanceBits.length) {
      answer =
        `Here is what the document says:\n\n` +
        substanceBits.map((t) => `• ${t}`).join('\n\n')
    }
  }

  return {
    answer,
    sources: sourceRefs.map((s) => s.label),
    sourceRefs,
  }
}

function cleanAnswerBody(raw: string): string {
  let text = raw.trim()
  // Drop trailing Sources / References blocks the model sometimes still adds
  text = text.replace(
    /\n+(?:#{1,3}\s*)?(?:\*{0,2})(?:Sources?|References?|Citations?|Page(?:s)?\s+links?)(?:\*{0,2})\s*:?\s*\n[\s\S]*$/i,
    '',
  )
  text = text.replace(/^\s*(?:Answer|ANSWER)\s*:\s*/i, '')
  return text.trim()
}

function extractMentionedPages(text: string): number[] {
  const found = new Set<number>()
  const re = /\b(?:page|p\.?)\s*(\d+)\b/gi
  let m: RegExpExecArray | null
  while ((m = re.exec(text)) !== null) {
    const n = Number(m[1])
    if (Number.isFinite(n) && n > 0 && n < 5000) found.add(n)
  }
  return [...found].sort((a, b) => a - b)
}
