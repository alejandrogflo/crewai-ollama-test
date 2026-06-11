from __future__ import annotations

from pathlib import Path

import pdfplumber
from crewai.tools import BaseTool


SUPPORTED_DOCUMENT_EXTENSIONS = {".md", ".txt", ".pdf"}


def is_supported_document(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_DOCUMENT_EXTENSIONS


def discover_candidate_files(directory: str | Path) -> list[Path]:
    directory_path = Path(directory).expanduser().resolve()
    if not directory_path.exists():
        raise FileNotFoundError(f"Candidate directory not found: {directory_path}")
    if not directory_path.is_dir():
        raise NotADirectoryError(f"Candidate path is not a directory: {directory_path}")

    files = sorted(
        path
        for path in directory_path.iterdir()
        if is_supported_document(path)
    )
    if not files:
        raise FileNotFoundError(
            f"No candidate documents found in {directory_path}. "
            f"Supported extensions: {', '.join(sorted(SUPPORTED_DOCUMENT_EXTENSIONS))}"
        )
    return files


def read_document_text(path: str | Path, max_chars: int = 12000) -> str:
    document_path = Path(path).expanduser().resolve()
    if not document_path.exists():
        raise FileNotFoundError(f"Document not found: {document_path}")

    suffix = document_path.suffix.lower()
    if suffix not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise ValueError(
            f"Unsupported document type {suffix}. "
            f"Use one of: {', '.join(sorted(SUPPORTED_DOCUMENT_EXTENSIONS))}"
        )

    if suffix == ".pdf":
        text_parts: list[str] = []
        with pdfplumber.open(document_path) as pdf:
            for page in pdf.pages:
                text_parts.append(page.extract_text() or "")
        text = "\n\n".join(text_parts).strip()
    else:
        text = document_path.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError(f"Document is empty: {document_path}")

    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[Truncated to fit context window]"

    return text


class ReadDocumentTool(BaseTool):
    name: str = "read_document"
    description: str = (
        "Read a local Markdown, text, or PDF document and return its text content. "
        "Use this when you only have a path and need evidence from the file."
    )

    def _run(self, path: str, max_chars: int = 12000) -> str:
        return read_document_text(path=path, max_chars=max_chars)


class ListCandidateFilesTool(BaseTool):
    name: str = "list_candidate_files"
    description: str = (
        "List candidate document files in a directory. "
        "Use this to discover which CV files are available for assessment."
    )

    def _run(self, directory: str) -> str:
        files = discover_candidate_files(directory)
        return "\n".join(str(path) for path in files)
