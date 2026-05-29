from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.configs import CONFIG
from src.tools.document_tools import _resolve_pdf_path, read_pdf


class DocumentToolsTests(unittest.TestCase):
    def setUp(self) -> None:
        CONFIG.workspace.root_dir.mkdir(parents=True, exist_ok=True)

    def test_resolve_pdf_path_accepts_container_workspace_path(self) -> None:
        expected = CONFIG.workspace.latest_dir / "resume.pdf"

        self.assertEqual(_resolve_pdf_path("/workspace/latest/resume.pdf"), expected)
        self.assertEqual(_resolve_pdf_path("workspace/latest/resume.pdf"), expected)

    def test_read_pdf_rejects_paths_outside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as outside_dir:
            outside_path = Path(outside_dir) / "resume.pdf"
            outside_path.write_bytes(b"%PDF-1.4\n")

            result = read_pdf.invoke({"file_path": str(outside_path)})

        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "PDF path must stay inside the workspace directory.")
        self.assertIsNone(_resolve_pdf_path(str(outside_path)))

    def test_read_pdf_maps_container_workspace_path_before_validation(self) -> None:
        with tempfile.TemporaryDirectory(dir=CONFIG.workspace.root_dir) as temp_dir:
            document_path = Path(temp_dir) / "resume.txt"
            document_path.write_text("not a pdf", encoding="utf-8")
            container_path = f"{CONFIG.docker.container_workspace_dir}/{document_path.relative_to(CONFIG.workspace.root_dir)}"

            result = read_pdf.invoke({"file_path": container_path})

        self.assertFalse(result["success"])
        self.assertEqual(result["file_path"], str(document_path))
        self.assertEqual(result["error"], "File is not a PDF.")

    def test_read_pdf_rejects_oversized_workspace_pdf_before_parsing(self) -> None:
        with tempfile.TemporaryDirectory(dir=CONFIG.workspace.root_dir) as temp_dir:
            document_path = Path(temp_dir) / "resume.pdf"
            document_path.write_bytes(b"%PDF-1.4\n123456")

            with patch("src.tools.document_tools.MAX_PDF_BYTES", 5):
                result = read_pdf.invoke({"file_path": str(document_path)})

        self.assertFalse(result["success"])
        self.assertIn("too large", result["error"])


if __name__ == "__main__":
    unittest.main()
