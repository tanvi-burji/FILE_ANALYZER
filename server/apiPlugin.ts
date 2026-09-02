import type { IncomingMessage, ServerResponse } from 'node:http'
import type { Plugin } from 'vite'

type Provider = 'openrouter' | 'groq' | 'gemini' | 'openai'

async function readBody(req: IncomingMessage): Promise<string> {
  const chunks: Buffer[] = []
  for await (const chunk of req) {
    chunks.push(typeof chunk === 'string' ? Buffer.from(chunk) : chunk)
  }
  return Buffer.concat(chunks).toString('utf8')
}

async function callGemini(apiKey: string, model: string, prompt: string): Promise<string> {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${encodeURIComponent(apiKey)}`
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      contents: [{ role: 'user', parts: [{ text: prompt }] }],
      generationConfig: { temperature: 0.15, maxOutputTokens: 8192 },
    }),
  })
  const data = (await res.json()) as {
    error?: { message?: string }
    candidates?: { content?: { parts?: { text?: string }[] } }[]
  }
  if (!res.ok) throw new Error(data.error?.message || `Gemini failed (${res.status})`)
  const text = data.candidates?.[0]?.content?.parts?.map((p) => p.text || '').join('') || ''
  if (!text.trim()) throw new Error('Empty model response')
  return text.trim()
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
  if (!res.ok) throw new Error(data.error?.message || `API failed (${res.status})`)
  const text = data.choices?.[0]?.message?.content || ''
  if (!text.trim()) throw new Error('Empty model response')
  return text.trim()
}

function sendJson(res: ServerResponse, status: number, body: unknown) {
  res.statusCode = status
  res.setHeader('Content-Type', 'application/json')
  res.end(JSON.stringify(body))
}

async function handleGenerate(req: IncomingMessage, res: ServerResponse) {
  if (req.method === 'OPTIONS') {
    res.statusCode = 204
    res.end()
    return
  }

  if (req.method !== 'POST') {
    sendJson(res, 405, { error: 'Method not allowed' })
    return
  }

  try {
    const raw = await readBody(req)
    const body = JSON.parse(raw) as {
      provider?: Provider
      apiKey?: string
      model?: string
      prompt?: string
    }

    const provider = body.provider || 'openrouter'
    const apiKey =
      body.apiKey ||
      process.env.VITE_OPENROUTER_API_KEY ||
      process.env.OPENROUTER_API_KEY ||
      process.env.VITE_GEMINI_API_KEY ||
      process.env.GEMINI_API_KEY ||
      process.env.VITE_GROQ_API_KEY ||
      process.env.GROQ_API_KEY ||
      process.env.VITE_OPENAI_API_KEY ||
      process.env.OPENAI_API_KEY ||
      ''

    if (!apiKey) {
      sendJson(res, 400, { error: 'Missing API key' })
      return
    }
    if (!body.prompt?.trim()) {
      sendJson(res, 400, { error: 'Missing prompt' })
      return
    }

    let text = ''
    if (provider === 'gemini') {
      text = await callGemini(apiKey, body.model || 'gemini-2.0-flash', body.prompt)
    } else if (provider === 'openrouter') {
      text = await callOpenAiCompatible(
        'https://openrouter.ai/api/v1',
        apiKey,
        body.model || 'google/gemini-2.0-flash-001',
        body.prompt,
        {
          'HTTP-Referer': 'http://localhost:5173',
          'X-Title': 'ScanAsk',
        },
      )
    } else if (provider === 'groq') {
      text = await callOpenAiCompatible(
        'https://api.groq.com/openai/v1',
        apiKey,
        body.model || 'llama-3.1-8b-instant',
        body.prompt,
      )
    } else {
      text = await callOpenAiCompatible(
        'https://api.openai.com/v1',
        apiKey,
        body.model || 'gpt-4o-mini',
        body.prompt,
      )
    }

    sendJson(res, 200, { text })
  } catch (err) {
    sendJson(res, 500, {
      error: err instanceof Error ? err.message : 'Generation failed',
    })
  }
}

function attach(middlewares: {
  use: (
    path: string,
    handler: (req: IncomingMessage, res: ServerResponse, next: () => void) => void,
  ) => void
}) {
  middlewares.use('/api/generate', (req, res, next) => {
    void handleGenerate(req, res).catch(next)
  })
}

export function scanaskApiPlugin(): Plugin {
  return {
    name: 'scanask-api',
    configureServer(server) {
      attach(server.middlewares)
    },
    configurePreviewServer(server) {
      attach(server.middlewares)
    },
  }
}
