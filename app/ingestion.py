"""Turn uploaded files into text chunks ready for the vector store."""

from io import BytesIO
from pathlib import PurePath

from docx import Document as DocxDocument
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pypdf.errors import PyPdfError

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


class UnsupportedDocumentError(ValueError):
    """The file can't be turned into text. The message is safe to show to users."""


def extract_pages(filename: str, data: bytes) -> list[tuple[int | None, str]]:
    """Return (page number, text) pairs. Formats without pages use None."""
    extension = PurePath(filename).suffix.lower()
    if extension == ".pdf":
        pages = _extract_pdf(data)
    elif extension == ".docx":
        pages = _extract_docx(data)
    elif extension in SUPPORTED_EXTENSIONS:
        pages = [(None, _decode_text(data))]
    else:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedDocumentError(f"Unsupported file type. Use one of: {supported}.")

    pages = [(number, text) for number, text in pages if text.strip()]
    if not pages:
        raise UnsupportedDocumentError(
            "No text found in this file. Scanned documents need OCR before uploading."
        )
    return pages


def chunk_document(
    filename: str, data: bytes, chunk_size: int, chunk_overlap: int
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = []
    for page, text in extract_pages(filename, data):
        metadata = {"filename": filename}
        if page is not None:
            metadata["page"] = page
        chunks.extend(splitter.create_documents([text], metadatas=[metadata]))
    return chunks


def _extract_pdf(data: bytes) -> list[tuple[int | None, str]]:
    try:
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted:
            raise UnsupportedDocumentError("Password-protected PDFs aren't supported.")
        return [
            (number, page.extract_text() or "")
            for number, page in enumerate(reader.pages, start=1)
        ]
    except PyPdfError as error:
        raise UnsupportedDocumentError("This PDF couldn't be read.") from error


def _extract_docx(data: bytes) -> list[tuple[int | None, str]]:
    try:
        document = DocxDocument(BytesIO(data))
    except Exception as error:  # python-docx raises many types for malformed zips
        raise UnsupportedDocumentError("This Word document couldn't be read.") from error

    lines = [paragraph.text for paragraph in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            lines.append(" | ".join(cell.text.strip() for cell in row.cells))
    return [(None, "\n".join(lines))]


def _decode_text(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1")
