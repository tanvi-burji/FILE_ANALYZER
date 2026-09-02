export type ProviderId = 'browser' | 'openrouter' | 'gemini' | 'groq' | 'openai'

const KEY_STORAGE = 'scanask.apiKey'
const PROVIDER_STORAGE = 'scanask.provider'

export function getStoredApiKey(): string {
  try {
    return localStorage.getItem(KEY_STORAGE)?.trim() ?? ''
  } catch {
    return ''
  }
}

export function getStoredProvider(): ProviderId {
  try {
    const value = localStorage.getItem(PROVIDER_STORAGE)
    if (
      value === 'browser' ||
      value === 'openrouter' ||
      value === 'gemini' ||
      value === 'openai' ||
      value === 'groq'
    ) {
      return value
    }
  } catch {
    // ignore
  }
  return 'browser'
}

export function setStoredCredentials(provider: ProviderId, key: string) {
  try {
    localStorage.setItem(PROVIDER_STORAGE, provider)
    if (provider === 'browser') {
      // Browser LLM needs no key
      return
    }
    if (key.trim()) localStorage.setItem(KEY_STORAGE, key.trim())
    else localStorage.removeItem(KEY_STORAGE)
  } catch {
    // ignore
  }
}

export function providerNeedsApiKey(provider: ProviderId): boolean {
  return provider !== 'browser'
}
