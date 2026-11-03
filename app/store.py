"""Chroma-backed document store, partitioned by workspace."""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

from fastembed import TextEmbedding
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class FastEmbedEmbeddings(Embeddings):
    """Local ONNX embeddings, so documents are never sent to an embedding API."""

    def __init__(self, model_name: str):
        self._model = TextEmbedding(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


@dataclass(frozen=True)
class DocumentInfo:
    id: str
    filename: str
    chunks: int


class DocumentStore:
    def __init__(self, vector_store: Chroma):
        self._vectors = vector_store

    def add(self, workspace: str, filename: str, data: bytes, chunks: Sequence[Document]) -> DocumentInfo:
        """Index a document. Uploading identical bytes again replaces the same entry."""
        document_id = hashlib.sha256(workspace.encode() + b"\0" + data).hexdigest()[:32]
        for chunk in chunks:
            chunk.metadata |= {"workspace": workspace, "document_id": document_id}
        self.delete(workspace, document_id)
        self._vectors.add_documents(
            list(chunks),
            ids=[f"{document_id}-{index}" for index in range(len(chunks))],
        )
        return DocumentInfo(document_id, filename, len(chunks))

    def list_documents(self, workspace: str) -> list[DocumentInfo]:
        result = self._vectors.get(where={"workspace": workspace}, include=["metadatas"])
        found: dict[str, DocumentInfo] = {}
        for metadata in result["metadatas"]:
            document_id = metadata["document_id"]
            known = found.get(document_id)
            found[document_id] = DocumentInfo(
                document_id, metadata["filename"], known.chunks + 1 if known else 1
            )
        return sorted(found.values(), key=lambda document: document.filename.lower())

    def delete(self, workspace: str, document_id: str) -> bool:
        result = self._vectors.get(where=_filter(workspace, document_id), include=[])
        if not result["ids"]:
            return False
        self._vectors.delete(ids=result["ids"])
        return True

    def search(
        self, workspace: str, query: str, limit: int, document_id: str | None = None
    ) -> list[Document]:
        return self._vectors.similarity_search(
            query, k=limit, filter=_filter(workspace, document_id)
        )


def _filter(workspace: str, document_id: str | None) -> dict:
    if document_id is None:
        return {"workspace": workspace}
    return {"$and": [{"workspace": workspace}, {"document_id": document_id}]}


def build_store(persist_directory: str, embedding_model: str) -> DocumentStore:
    return DocumentStore(
        Chroma(
            collection_name="documents",
            embedding_function=FastEmbedEmbeddings(embedding_model),
            persist_directory=persist_directory,
            collection_metadata={"hnsw:space": "cosine"},
        )
    )
