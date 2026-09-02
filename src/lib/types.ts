export type ChatRole = 'user' | 'assistant'

export interface SourceRef {
  page: number
  title: string
  label: string
  /** Supporting statement / quote from that page */
  snippet?: string
}

export interface ChatMessage {
  id: string
  role: ChatRole
  content: string
  sources?: string[]
  sourceRefs?: SourceRef[]
}

export interface DocumentPage {
  page: number
  title: string
  text: string
}

export interface AnalyzedFile {
  id: string
  name: string
  type: string
  size: number
  text: string
  pages: DocumentPage[]
  pageCount: number
  wordCount: number
  preview: string
  imageCount: number
  status: 'extracting' | 'training' | 'ready' | 'error'
  error?: string
}

export interface KnowledgeBrief {
  summary: string
  documentType: string
  topics: string[]
  entities: { name: string; role?: string }[]
  keyFacts: string[]
  dates: string[]
  sections: { title: string; gist: string; page?: number }[]
}

export interface DocumentBrain {
  brief: KnowledgeBrief
  fileNames: string[]
  trainedAt: number
}
