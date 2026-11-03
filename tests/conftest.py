import json
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_chroma import Chroma
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

from app.agent import DocumentAgent
from app.config import Settings, get_settings
from app.main import app
from app.store import DocumentStore


class ScriptedChatModel(GenericFakeChatModel):
    """Plays back a fixed list of responses and accepts tool binding.

    Each response is sent as one chunk, including tool calls and empty answers.
    """

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        return self

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        message = next(self.messages)
        tool_call_chunks = [
            {
                "name": call["name"],
                "args": json.dumps(call["args"]),
                "id": call["id"],
                "index": index,
            }
            for index, call in enumerate(message.tool_calls)
        ]
        yield ChatGenerationChunk(
            message=AIMessageChunk(content=message.content, tool_call_chunks=tool_call_chunks)
        )


def scripted_model(*messages: AIMessage) -> ScriptedChatModel:
    return ScriptedChatModel(messages=iter(messages))


@pytest.fixture
def store(request) -> DocumentStore:
    return DocumentStore(
        Chroma(
            collection_name=f"test-{id(request)}",
            embedding_function=DeterministicFakeEmbedding(size=32),
        )
    )


@pytest.fixture
def make_client(store, tmp_path):
    """Build clients that don't share cookies, so each one is a separate workspace."""
    settings = Settings(_env_file=None, chroma_dir=tmp_path / "chroma")
    app.dependency_overrides[get_settings] = lambda: settings

    def make(model: ScriptedChatModel | None = None) -> TestClient:
        client = TestClient(app)
        client.__enter__()
        app.state.store = store
        app.state.agent = DocumentAgent(model, store, 4) if model else None
        return client

    yield make
    app.dependency_overrides.clear()
