https://gdkservices.co.uk

# Document Q&A Agent

Upload documents, then ask questions about them. A Claude agent searches the documents, can search again when the first results aren't enough, and answers with citations.

Built with FastAPI, LangChain, and Chroma. Embeddings run locally with FastEmbed, so document text only reaches Anthropic when the agent retrieves it to answer a question.

## Run it

Requires Python 3.12 and an [Anthropic API key](https://console.anthropic.com/).

```bash
python3 -m venv .venv
```

```bash
.venv/bin/pip install -r requirements.txt
```

```bash
cp .env.example .env
```

Add your key to `.env`, then start the server:

```bash
.venv/bin/uvicorn app.main:app --reload
```

Open <http://localhost:8000>. The first start downloads the embedding model (about 70 MB).

## How it works

```
Upload  →  extract text  →  split into chunks  →  embed locally  →  Chroma
Question  →  Claude agent  ⇄  list_documents / search_documents  →  cited answer
```

| File | Purpose |
| --- | --- |
| `app/ingestion.py` | Extracts text from PDF, Word, text, and Markdown files, and splits it into chunks |
| `app/store.py` | Stores and searches chunks in Chroma, partitioned by workspace |
| `app/agent.py` | The LangChain agent, its tools, and its system prompt |
| `app/main.py` | FastAPI routes and workspace cookies |
| `app/static/index.html` | The web UI |
| `app/static/how-it-works.html` | Animated explainer of the architecture, served at `/how-it-works` |

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/documents` | Upload a file (multipart field `file`) |
| `GET` | `/api/documents` | List your documents |
| `DELETE` | `/api/documents/{id}` | Delete a document |
| `POST` | `/api/chat` | Ask a question and get the full answer as JSON |
| `POST` | `/api/chat/stream` | Ask a question and stream the answer as server-sent events |

Chat requests take `question` and an optional `conversation_id`. Send back the returned ID to continue a conversation. Interactive docs are at `/docs`.

With curl, keep the workspace cookie between requests:

```bash
curl -c cookies.txt -b cookies.txt -F file=@report.pdf http://localhost:8000/api/documents
```

```bash
curl -c cookies.txt -b cookies.txt -H 'content-type: application/json' -d '{"question":"What are the key findings?"}' http://localhost:8000/api/chat
```

## Limits

- **No accounts.** Each browser gets a private workspace through a random cookie. Clearing cookies loses access to its documents, and anyone holding the cookie can read them. Add real authentication before storing sensitive data on a public server, and set `COOKIE_SECURE=true` behind HTTPS.
- **No rate limiting.** Every question spends API credit, so a public deployment needs limits on uploads and chat.
- **Conversations are kept in memory.** They're lost on restart and grow until then. Documents persist in `data/chroma`.
- **One server process.** Conversations aren't shared between workers, so run a single worker or replace `InMemorySaver` with a shared LangGraph checkpointer.
- **Text only.** Scanned PDFs need OCR first, and images and charts are ignored.

## Tests

```bash
.venv/bin/python -m pytest
```

Tests use a scripted model and fake embeddings, so they need no API key or network.
