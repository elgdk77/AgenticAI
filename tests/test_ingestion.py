from io import BytesIO

import pytest
from docx import Document
from pypdf import PdfWriter

from app.ingestion import UnsupportedDocumentError, chunk_document, extract_pages


def test_text_file_is_split_with_filename_metadata():
    chunks = chunk_document("notes.txt", ("word " * 500).encode(), 200, 20)

    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= 200 for chunk in chunks)
    assert all(chunk.metadata == {"filename": "notes.txt"} for chunk in chunks)


def test_docx_includes_paragraphs_and_tables():
    document = Document()
    document.add_paragraph("Quarterly summary")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Revenue"
    table.cell(0, 1).text = "42"
    buffer = BytesIO()
    document.save(buffer)

    [(page, text)] = extract_pages("report.docx", buffer.getvalue())

    assert page is None
    assert "Quarterly summary" in text
    assert "Revenue | 42" in text


def test_non_utf8_text_is_still_decoded():
    assert extract_pages("old.txt", "café".encode("latin-1")) == [(None, "café")]


def test_unsupported_extension_is_rejected():
    with pytest.raises(UnsupportedDocumentError, match="Unsupported file type"):
        extract_pages("archive.zip", b"PK")


def test_blank_file_is_rejected():
    with pytest.raises(UnsupportedDocumentError, match="No text found"):
        extract_pages("blank.txt", b"  \n ")


def test_pdf_without_text_is_rejected():
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = BytesIO()
    writer.write(buffer)

    with pytest.raises(UnsupportedDocumentError, match="OCR"):
        extract_pages("scan.pdf", buffer.getvalue())


def test_corrupt_files_are_rejected():
    with pytest.raises(UnsupportedDocumentError):
        extract_pages("broken.pdf", b"not a pdf")
    with pytest.raises(UnsupportedDocumentError):
        extract_pages("broken.docx", b"not a docx")


def test_pdf_text_keeps_page_numbers():
    pdf = b"""%PDF-1.4
1 0 obj <</Type/Catalog/Pages 2 0 R>> endobj
2 0 obj <</Type/Pages/Kids[3 0 R 5 0 R]/Count 2>> endobj
3 0 obj <</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]/Contents 4 0 R
/Resources<</Font<</F1 7 0 R>>>>>> endobj
4 0 obj <</Length 37>> stream
BT /F1 12 Tf 20 100 Td (First) Tj ET
endstream endobj
5 0 obj <</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]/Contents 6 0 R
/Resources<</Font<</F1 7 0 R>>>>>> endobj
6 0 obj <</Length 38>> stream
BT /F1 12 Tf 20 100 Td (Second) Tj ET
endstream endobj
7 0 obj <</Type/Font/Subtype/Type1/BaseFont/Helvetica>> endobj
trailer <</Root 1 0 R>>
startxref
0
%%EOF
"""
    chunks = chunk_document("deck.pdf", pdf, 200, 20)

    assert [(chunk.page_content, chunk.metadata["page"]) for chunk in chunks] == [
        ("First", 1),
        ("Second", 2),
    ]
