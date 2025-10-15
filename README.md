# Agentic LLM Application with Tool-Use & Memory

A full-stack, agentic LLM application featuring advanced tool-use, in-session memory, and a modern React/Tailwind UI. Designed for robust, real-world AI assistant scenarios.

---

## Features

- **Agentic LLM Backend**: Python FastAPI server using LangChain, LangGraph, and Ollama for LLM orchestration and tool chaining.
- **Tool-Use & Function Calling**: Supports file I/O, calculator, web search, stock price, news headlines, weather, and currency conversion tools.
- **Multi-Step Planning**: Handles complex, multi-tool queries and can return structured JSON outputs for downstream consumption.
- **Memory**: Maintains in-session conversation history; easily extendable to persistent vector DB (Chroma).
- **Modern UI**: React + Vite + Tailwind CSS frontend with avatars, markdown rendering, and mobile-friendly design.
- **Logging**: All tool calls are logged for transparency and debugging.

---

## Tech Stack

- **Backend**: Python, FastAPI, LangChain, LangGraph, Ollama
- **Frontend**: React, TypeScript, Vite, Tailwind CSS
- **APIs**: Alpha Vantage (stocks), NewsAPI (news), Open-Meteo (weather), Google Custom Search (web)

---

## Quickstart

### 1. Backend (Python)

```bash
# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn main:app --reload
```

- The backend runs on `http://localhost:8000`
- Requires a `.env` file with API keys:
  - `ALPHA_VANTAGE_KEY`, `NEWSAPI_KEY`, `GOOGLE_CUSTOM_SEARCH_API_KEY`, `SEARCH_ENGINE_ID`

### 2. Frontend (React)

```bash
cd app_ui
npm install
npm run dev
```

- The frontend runs on `http://localhost:5173` (default Vite port)

---

## Docker Compose (Recommended)

To run the entire stack (Ollama, backend, frontend) with one command:

```bash
docker compose up -d
```

- Ollama (LLM server) runs on port 11434
- Backend (FastAPI) runs on port 8000
- Frontend (React) runs on port 5173

**Important:** After starting the stack for the first time, you must pull the llama3.2 model into the Ollama container:

```bash
docker exec -it agentic-llm-application-with-tool-use-memory-ollama-1 ollama pull llama3.2
```

This only needs to be done once, unless you remove the Ollama volume.

You can stop all services with:

```bash
docker compose down
```

Make sure you have a `.env` file in the root directory with all required API keys before starting.

---

## Usage

- Chat with the assistant in the web UI.
- Try queries like:
  - `What is the weather in London?`
  - `Get the latest news about Apple as JSON.`
  - `Calculate 2 * (3 + 4)`
  - `List files in the current directory.`
  - `Convert 100 USD to EUR and GBP.`
- The assistant can chain tools and return structured JSON if requested.
- Stock prices use Alpha Vantage (free, but rate-limited; always in USD).
- News headlines use NewsAPI (free tier is rate-limited; requires API key).
- Weather data uses Open-Meteo (no API key required; free and open).
- Web search uses Google Custom Search (requires API key and search engine ID; subject to quota and billing).
- All tool calls are logged in the backend for debugging.

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

---

## Frontend Highlights

- Modern, mobile-friendly chat UI (React, Tailwind)
- Markdown rendering with code highlighting
- Avatars for user and assistant
- Error handling for failed API calls

---

## Environment Variables

Create a `.env` file in the root directory with:

```
ALPHA_VANTAGE_KEY=your_alpha_vantage_key
NEWSAPI_KEY=your_newsapi_key
GOOGLE_CUSTOM_SEARCH_API_KEY=your_google_key
SEARCH_ENGINE_ID=your_search_engine_id
```

---

## License

MIT
