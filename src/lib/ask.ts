import { askWithBrain, buildLocalIndex } from './llm'
import type { AnalyzedFile, ChatMessage, DocumentBrain, SourceRef } from './types'

/** Index the file from its text (no model download). */
export async function indexDocument(file: AnalyzedFile): Promise<DocumentBrain> {
  return buildLocalIndex([file], { provider: 'browser', apiKey: '', model: '' })
}

export async function answerQuestion(
  question: string,
  files: AnalyzedFile[],
  brain: DocumentBrain | null,
  history: ChatMessage[] = [],
): Promise<{ answer: string; sources: string[]; sourceRefs: SourceRef[] }> {
  return askWithBrain(
    question,
    files.filter((f) => f.text.trim()),
    brain,
    { provider: 'browser', apiKey: '', model: '' },
    history
      .filter((m) => m.role === 'user' || m.role === 'assistant')
      .map((m) => ({ role: m.role, content: m.content })),
  )
}
