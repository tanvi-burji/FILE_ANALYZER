/** Extract page numbers from answer/source text. */
export function extractPageNumbers(text: string): number[] {
  const found = new Set<number>()
  const patterns = [
    /\bpages?\s*[#:]?\s*(\d+)\b/gi,
    /\bp\.\s*(\d+)\b/gi,
    /\bp\s+(\d+)\b/gi,
    /\[\[\[PAGE\s+(\d+)/gi,
    /----- PAGE\s+(\d+)/gi,
  ]

  for (const re of patterns) {
    let match: RegExpExecArray | null
    while ((match = re.exec(text)) !== null) {
      const n = Number(match[1])
      if (Number.isFinite(n) && n > 0 && n < 5000) found.add(n)
    }
  }

  return [...found].sort((a, b) => a - b)
}

/** Markdown-lite HTML for the answer body (no page chips — those render separately). */
export function renderAnswerHtml(text: string): string {
  let html = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')

  // Light bullets / numbered steps
  html = html.replace(/(^|\n)[-•]\s+/g, '$1• ')
  html = html.replace(/(^|\n)(\d+)\.\s+/g, '$1$2. ')

  return html.replace(/\n\n/g, '<br/><br/>').replace(/\n/g, '<br/>')
}
