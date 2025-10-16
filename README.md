# Agentic LLM Application with Tool-Use & Memory

A full-stack AI assistant featuring advanced tool-use, in-session memory, document retrieval (RAG), and a modern React/Tailwind UI. Built for robust, real-world agentic scenarios.

---

## Features

- **Agentic LLM Backend**: FastAPI server orchestrating LangChain, LangGraph, and Ollama for multi-step reasoning and tool chaining.
- **Tool-Use & Function Calling**: File I/O, calculator, web search, stock price, news headlines, weather, currency conversion, input sanitization.
- **Memory**: Maintains in-session conversation history; extendable to persistent vector DB (Chroma).
- **Document RAG**: PDF/CSV ingestion, ensemble retrieval, and ChromaDB hybrid search.
- **Modern UI**: React + Vite + Tailwind CSS frontend with avatars, markdown rendering, and mobile-friendly design.
- **Logging**: All tool calls are logged for transparency and debugging.

---

## Tech Stack

- **Backend**: Python, FastAPI, LangChain, LangGraph, Ollama, ChromaDB
- **Frontend**: React, TypeScript, Vite, Tailwind CSS
- **APIs**: Alpha Vantage (stocks), NewsAPI (news), Open-Meteo (weather), Google Custom Search (web)

---

## Quickstart

### Backend (Python)

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

- Runs on `http://localhost:8000`
- Requires `.env` file with API keys:
  - `ALPHA_VANTAGE_KEY`, `NEWSAPI_KEY`, `GOOGLE_CUSTOM_SEARCH_API_KEY`, `SEARCH_ENGINE_ID`

### Frontend (React)

```bash
cd app_ui
npm install
npm run dev
```

- Runs on `http://localhost:5173`

---

## Docker Compose

To run the entire stack (Ollama, backend, frontend):

```bash
docker compose up -d
```

- Ollama (LLM server): port 11434
- Backend (FastAPI): port 8000
- Frontend (React): port 5173

Stop all services:

```bash
docker compose down
```

---

## Usage

- Chat with the assistant in the web UI.
- Example queries:
  - `What is the weather in London?`
  - `Search the web for Albert Einstein`
  - `Calculate 2 * (3 + 4)`
  - `List files in the current directory.`
  - `Convert 100 USD to EUR and GBP.`
  - `Get the name of the student from the uploaded document`
  - `Write Hello World! to temp.txt`
  - `Read contents from story.txt`
  - `What is the time right now?`
- The assistant chains tools and returns structured JSON if requested.
- All tool calls are logged for debugging.

---

## Backend Endpoints

- `POST /chat` — Main chat endpoint. Request body:
  ```json
  {
    "message": "Your question or command",
    "structured_output": false
  }
  ```
  Returns: `{ "response": ... }`
- `POST /upload-file` — Upload PDF/CSV files for RAG.
- `POST /create-vector-database` — Ingest uploaded files into ChromaDB.

---

## Frontend Highlights

- Modern, mobile-friendly chat UI (React, Tailwind)
- Markdown rendering with code highlighting
- Avatars for user and assistant
- Error handling for failed API calls

---

## Environment Variables

Create a `.env` file in the root directory:

```
ALPHA_VANTAGE_KEY=your_alpha_vantage_key
NEWSAPI_KEY=your_newsapi_key
GOOGLE_CUSTOM_SEARCH_API_KEY=your_google_key
SEARCH_ENGINE_ID=your_search_engine_id
```

---

## License

MIT
