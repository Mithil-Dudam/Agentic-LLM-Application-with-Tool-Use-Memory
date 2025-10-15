# Agentic LLM Frontend (app_ui)

This is the frontend for the Agentic LLM Application with Tool-Use & Memory. It is built with React, TypeScript, Vite, and Tailwind CSS, and provides a modern, mobile-friendly chat interface for interacting with the backend AI agent.

## Features

- Clean chat UI with avatars for user and assistant
- Markdown rendering (including code blocks)
- Responsive/mobile-friendly design
- Error handling for failed API calls
- Supports structured JSON output from the backend

## Getting Started

### Development

1. Install dependencies:
   ```bash
   npm install
   ```
2. Start the development server:
   ```bash
   npm run dev
   ```
   The app will be available at `http://localhost:5173` by default.

### Production Build

To build the frontend for production:

```bash
npm run build
```

The static files will be output to the `dist/` directory. In Docker Compose, these are served by Nginx.

## Connecting to the Backend

The frontend expects the backend API to be available at `/chat` (proxied or CORS-enabled). When using Docker Compose, the frontend and backend are automatically networked together.

## Project Structure

- `src/` — Main React source code
- `src/components/` — UI components (Avatar, MarkdownMessage, etc.)
- `src/pages/` — Page components (Home, PageNotFound)
- `public/` — Static assets

## Customization

You can modify the UI, add new features, or change the theme by editing the React components and Tailwind CSS classes in `src/`.

---

For more details, see the main project [README.md](../README.md).
