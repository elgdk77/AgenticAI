import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path, PurePath
from typing import Annotated
from uuid import UUID, uuid4

import anthropic
from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.agent import AnswerSource, AnswerText, DocumentAgent, EmptyAnswerError, build_model
from app.config import Settings, get_settings
from app.ingestion import UnsupportedDocumentError, chunk_document
from app.store import DocumentInfo, DocumentStore, build_store

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
WORKSPACE_COOKIE = "workspace_id"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    app.state.store = build_store(str(settings.chroma_dir), settings.embedding_model)
    app.state.agent = None
    if settings.anthropic_api_key:
        app.state.agent = DocumentAgent(
            build_model(settings), app.state.store, settings.search_results
        )
    yield


app = FastAPI(title="Document Q&A Agent", lifespan=lifespan)


@app.middleware("http")
async def assign_workspace(request: Request, call_next):
    """Give each browser a private workspace, identified by an unguessable cookie."""
    cookie = request.cookies.get(WORKSPACE_COOKIE)
    workspace = cookie if cookie and _is_uuid(cookie) else None
    request.state.workspace = workspace or str(uuid4())

    response = await call_next(request)
    if workspace is None:
        response.set_cookie(
            WORKSPACE_COOKIE,
            request.state.workspace,
            max_age=60 * 60 * 24 * 365,
            httponly=True,
            samesite="lax",
            secure=get_settings().cookie_secure,
        )
    return response


def _is_uuid(value: str) -> bool:
    try:
        UUID(value)
    except ValueError:
        return False
    return True


def get_workspace(request: Request) -> str:
    return request.state.workspace


def get_store(request: Request) -> DocumentStore:
    return request.app.state.store


def get_agent(request: Request) -> DocumentAgent:
    agent = request.app.state.agent
    if agent is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "ANTHROPIC_API_KEY is not set on the server.",
        )
    return agent


Workspace = Annotated[str, Depends(get_workspace)]
Store = Annotated[DocumentStore, Depends(get_store)]
Agent = Annotated[DocumentAgent, Depends(get_agent)]
AppSettings = Annotated[Settings, Depends(get_settings)]


class DocumentResponse(BaseModel):
    id: str
    filename: str
    chunks: int

    @classmethod
    def of(cls, document: DocumentInfo) -> "DocumentResponse":
        return cls(id=document.id, filename=document.filename, chunks=document.chunks)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    conversation_id: str = Field(default_factory=lambda: uuid4().hex, max_length=64)


class SourceResponse(BaseModel):
    filename: str
    page: int | None
    excerpt: str


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    sources: list[SourceResponse]


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/documents", status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile, workspace: Workspace, store: Store, settings: AppSettings
) -> DocumentResponse:
    max_bytes = settings.max_upload_mb * 1024 * 1024
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"Files can be at most {settings.max_upload_mb} MB.",
        )

    # The name is only ever displayed, but drop any client-supplied path.
    filename = PurePath((file.filename or "document").replace("\\", "/")).name

    def ingest() -> DocumentInfo:
        chunks = chunk_document(filename, data, settings.chunk_size, settings.chunk_overlap)
        return store.add(workspace, filename, data, chunks)

    try:
        document = await run_in_threadpool(ingest)
    except UnsupportedDocumentError as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from error
    return DocumentResponse.of(document)


@app.get("/api/documents")
async def list_documents(workspace: Workspace, store: Store) -> list[DocumentResponse]:
    documents = await run_in_threadpool(store.list_documents, workspace)
    return [DocumentResponse.of(document) for document in documents]


@app.delete("/api/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: str, workspace: Workspace, store: Store) -> None:
    if not await run_in_threadpool(store.delete, workspace, document_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found.")


def _chat_error(error: Exception) -> tuple[int, str]:
    """Map an agent failure to a status code and a message safe to show to users."""
    if isinstance(error, EmptyAnswerError):
        return status.HTTP_502_BAD_GATEWAY, "The model returned no answer. Try rephrasing."
    if isinstance(error, anthropic.AuthenticationError):
        return status.HTTP_503_SERVICE_UNAVAILABLE, "The server's Anthropic API key was rejected."
    if isinstance(error, anthropic.RateLimitError):
        return status.HTTP_429_TOO_MANY_REQUESTS, "Rate limited. Try again shortly."
    if isinstance(error, anthropic.APIError):
        return status.HTTP_502_BAD_GATEWAY, "The model request failed. Try again."
    raise error


@app.post("/api/chat")
async def chat(body: ChatRequest, workspace: Workspace, agent: Agent) -> ChatResponse:
    answer: list[str] = []
    sources: list[SourceResponse] = []
    try:
        async for event in agent.stream(workspace, body.conversation_id, body.question):
            if isinstance(event, AnswerText):
                answer.append(event.text)
            else:
                sources.append(SourceResponse(**asdict(event)))
    except (EmptyAnswerError, anthropic.APIError) as error:
        logger.exception("Chat request failed")
        status_code, message = _chat_error(error)
        raise HTTPException(status_code, message) from error
    return ChatResponse(
        conversation_id=body.conversation_id, answer="".join(answer), sources=sources
    )


@app.post("/api/chat/stream")
async def chat_stream(body: ChatRequest, workspace: Workspace, agent: Agent) -> StreamingResponse:
    """Stream the answer as server-sent events: `text`, `source`, `error`, and `done`."""

    def sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    async def events() -> AsyncIterator[str]:
        try:
            async for event in agent.stream(workspace, body.conversation_id, body.question):
                if isinstance(event, AnswerSource):
                    yield sse("source", asdict(event))
                else:
                    yield sse("text", {"text": event.text})
        except (EmptyAnswerError, anthropic.APIError) as error:
            logger.exception("Chat stream failed")
            yield sse("error", {"message": _chat_error(error)[1]})
        yield sse("done", {"conversation_id": body.conversation_id})

    return StreamingResponse(events(), media_type="text/event-stream")
