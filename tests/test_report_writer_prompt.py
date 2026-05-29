from __future__ import annotations

import unittest

from src.sub_agents.report_writer import REPORT_WRITER_SYSTEM_PROMPT


class ReportWriterPromptTests(unittest.TestCase):
    def test_report_writer_preserves_urls_for_unverified_and_backlog_jobs(self) -> None:
        self.assertIn("Good But Unverified", REPORT_WRITER_SYSTEM_PROMPT)
        self.assertIn("Backlog To Verify Later", REPORT_WRITER_SYSTEM_PROMPT)
        self.assertIn("include a URL column or URL", REPORT_WRITER_SYSTEM_PROMPT)
        self.assertIn("Do not write \"URL not supplied\"", REPORT_WRITER_SYSTEM_PROMPT)
        self.assertIn("apply_url", REPORT_WRITER_SYSTEM_PROMPT)
        self.assertIn("final_url", REPORT_WRITER_SYSTEM_PROMPT)
        self.assertIn("canonical_url", REPORT_WRITER_SYSTEM_PROMPT)
        self.assertIn("first source_urls entry", REPORT_WRITER_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
