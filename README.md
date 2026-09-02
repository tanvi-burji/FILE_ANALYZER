# ScanAsk

Upload a PDF, DOCX, or text file. A **browser LLM** trains on it locally — no API key required.

## Run

```bash
cd scanask
npm install
npm run dev
```

Open the URL Vite prints. Use **Chrome or Edge** (WebGPU required).

## How it works

1. Extracts text from your file in the browser
2. Downloads a local model once (`Llama-3.2-3B`, ~2GB) via WebLLM
3. Trains on document chunks and answers questions on-device

Your file never leaves the machine when using Browser LLM.

## Optional cloud providers

In **Settings** you can still switch to OpenRouter / Gemini / Groq / OpenAI if you prefer a cloud model.
