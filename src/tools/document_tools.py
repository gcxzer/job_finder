from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.tools import tool
from pypdf import PdfReader

from src.configs import CONFIG

WORKSPACE_DIR = CONFIG.workspace.root_dir
CONTAINER_WORKSPACE = CONFIG.docker.container_workspace_dir
MAX_PDF_BYTES = 10_000_000
MAX_PDF_PAGES = 20
MAX_PDF_TEXT_CHARS = 120_000


@tool
def read_pdf(file_path: str) -> dict[str, Any]:
    """Read text from a local PDF file path."""
    path = _resolve_pdf_path(file_path)
    if path is None:
        return {
            "success": False,
            "file_path": file_path,
            "error": "PDF path must stay inside the workspace directory.",
        }
    if not path.exists():
        return {
            "success": False,
            "file_path": str(path),
            "error": "PDF file does not exist.",
        }
    if not path.is_file():
        return {
            "success": False,
            "file_path": str(path),
            "error": "PDF path is not a file.",
        }
    if path.suffix.lower() != ".pdf":
        return {
            "success": False,
            "file_path": str(path),
            "error": "File is not a PDF.",
        }
    if path.stat().st_size > MAX_PDF_BYTES:
        return {
            "success": False,
            "file_path": str(path),
            "error": f"PDF file is too large. Maximum size is {MAX_PDF_BYTES} bytes.",
        }

    try:
        reader = PdfReader(str(path))
        if len(reader.pages) > MAX_PDF_PAGES:
            return {
                "success": False,
                "file_path": str(path),
                "page_count": len(reader.pages),
                "error": f"PDF has too many pages. Maximum page count is {MAX_PDF_PAGES}.",
            }
        pages: list[str] = []
        page_errors: list[str] = []

        for page_number, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
            except Exception as error:  # pypdf can raise parser-specific errors.
                page_errors.append(f"page {page_number}: {error}")
                continue
            if page_text.strip():
                pages.append(page_text.strip())

        text = "\n\n".join(pages).strip()
        text_truncated = len(text) > MAX_PDF_TEXT_CHARS
        if text_truncated:
            text = text[:MAX_PDF_TEXT_CHARS].rstrip()
        return {
            "success": True,
            "file_path": str(path),
            "page_count": len(reader.pages),
            "text": text,
            "text_truncated": text_truncated,
            "warnings": page_errors,
        }
    except Exception as error:
        return {
            "success": False,
            "file_path": str(path),
            "error": str(error),
        }


def _resolve_pdf_path(file_path: str) -> Path | None:
    clean_path = str(file_path or "").strip()
    container_workspace = CONTAINER_WORKSPACE.rstrip("/")

    if clean_path == container_workspace:
        path = WORKSPACE_DIR
    elif clean_path.startswith(f"{container_workspace}/"):
        path = WORKSPACE_DIR / clean_path[len(container_workspace) :].lstrip("/")
    elif clean_path == "workspace":
        path = WORKSPACE_DIR
    elif clean_path.startswith("workspace/"):
        path = WORKSPACE_DIR / clean_path[len("workspace/") :].lstrip("/")
    else:
        raw_path = Path(clean_path).expanduser()
        path = raw_path if raw_path.is_absolute() else WORKSPACE_DIR / clean_path.lstrip("/")

    resolved = path.resolve()
    try:
        resolved.relative_to(WORKSPACE_DIR.resolve())
    except ValueError:
        return None
    return resolved


DOCUMENT_TOOLS = [read_pdf]
