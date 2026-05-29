from __future__ import annotations

import unittest

from src.sub_agents.report_writer import REPORT_WRITER_SYSTEM_PROMPT


class ReportWriterPromptTests(unittest.TestCase):
    def test_report_writer_preserves_urls_for_unverified_and_backlog_jobs(self) -> None:
        self.assertIn("Good But Unverified", REPORT_WRITER_SYSTEM_PROMPT)
        self.assertIn("Backlog To Verify Later", REPORT_WRITER_SYSTEM_PROMPT)
        self.assertIn("use a Markdown table", REPORT_WRITER_SYSTEM_PROMPT)
        self.assertIn("columns: Role, Company, Verification status, Recommendation, URL", REPORT_WRITER_SYSTEM_PROMPT)
        self.assertIn("list each not_verified_backlog job as its own table", REPORT_WRITER_SYSTEM_PROMPT)
        self.assertIn("Do not group multiple jobs into one bullet or summary row", REPORT_WRITER_SYSTEM_PROMPT)
        self.assertIn("Do not write \"URL not supplied\"", REPORT_WRITER_SYSTEM_PROMPT)
        self.assertIn("apply_url", REPORT_WRITER_SYSTEM_PROMPT)
        self.assertIn("final_url", REPORT_WRITER_SYSTEM_PROMPT)
        self.assertIn("canonical_url", REPORT_WRITER_SYSTEM_PROMPT)
        self.assertIn("first source_urls entry", REPORT_WRITER_SYSTEM_PROMPT)

    def test_report_writer_constrains_best_targets_to_non_skip_recommendations(self) -> None:
        self.assertIn("Best Verified Targets may include only jobs", REPORT_WRITER_SYSTEM_PROMPT)
        self.assertIn("recommendation is Apply or Maybe", REPORT_WRITER_SYSTEM_PROMPT)
        self.assertIn("Do not put Skip jobs in", REPORT_WRITER_SYSTEM_PROMPT)
        self.assertIn(
            "columns: Role, Company,\n  Verification status, Recommendation, Match score, URL, Notes",
            REPORT_WRITER_SYSTEM_PROMPT,
        )

    def test_report_writer_does_not_overgeneralize_recommendations(self) -> None:
        self.assertIn("derive counts from Match State JSON", REPORT_WRITER_SYSTEM_PROMPT)
        self.assertIn("Do not say all recommendations are Maybe", REPORT_WRITER_SYSTEM_PROMPT)
        self.assertIn("unless every reviewed job has that recommendation", REPORT_WRITER_SYSTEM_PROMPT)
        self.assertIn("Recommendation\n  cells must contain only Apply, Maybe, Skip, or Unspecified", REPORT_WRITER_SYSTEM_PROMPT)
        self.assertIn("Do not introduce new recommendation labels", REPORT_WRITER_SYSTEM_PROMPT)
        self.assertIn("Conditional", REPORT_WRITER_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
