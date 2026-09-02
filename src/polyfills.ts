// pdf.js v6 expects Math.sumPrecise (newer Chrome). Polyfill for older runtimes.
if (typeof Math.sumPrecise !== 'function') {
  Object.defineProperty(Math, 'sumPrecise', {
    value(numbers: Iterable<number>) {
      let sum = 0
      for (const n of numbers) sum += n
      return sum
    },
    writable: true,
    configurable: true,
  })
}

export {}
