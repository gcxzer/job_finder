from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from langchain_core.messages import HumanMessage

from src import task_runner


class TaskRunnerLoggingTests(unittest.TestCase):
    def test_run_task_logs_non_ai_messages_without_tool_call_attribute(self) -> None:
        class FakeAgent:
            async def astream(self, *args: object, **kwargs: object):
                yield {"node": {"messages": [HumanMessage(content="hello")]}}

        with patch.object(task_runner, "logger") as logger:
            asyncio.run(task_runner.run_task("target_roles: Backend Engineer", FakeAgent()))

        self.assertTrue(logger.info.called)
        logger.exception.assert_not_called()

    def test_safe_log_text_redacts_and_truncates_sensitive_content(self) -> None:
        text = (
            "Candidate email jane@example.com phone +49 151 12345678. "
            "Search date 2026-05-28. "
            + "resume details " * 200
        )

        result = task_runner._safe_log_text(text, max_chars=180)

        self.assertIn("[REDACTED_EMAIL]", result)
        self.assertIn("[REDACTED_PHONE]", result)
        self.assertIn("2026-05-28", result)
        self.assertIn("truncated", result)
        self.assertNotIn("jane@example.com", result)

    def test_safe_log_text_preserves_salary_ranges_and_year_series(self) -> None:
        text = "Salary 60.000-80.000 EUR, alternative 60000-80000 EUR. Years 2020 2021 2022."

        result = task_runner._safe_log_text(text, max_chars=1000)

        self.assertIn("60.000-80.000", result)
        self.assertIn("60000-80000", result)
        self.assertIn("2020 2021 2022", result)
        self.assertNotIn("[REDACTED_PHONE]", result)

    def test_safe_log_value_truncates_nested_tool_arguments(self) -> None:
        result = task_runner._safe_log_value({"content": "x" * 1300, "email": "jane@example.com"})

        self.assertIn("truncated", result["content"])
        self.assertEqual(result["email"], "[REDACTED_EMAIL]")

    def test_agent_config_raises_recursion_limit_for_full_pipeline(self) -> None:
        self.assertGreaterEqual(task_runner.agent_config["recursion_limit"], 80)


if __name__ == "__main__":
    unittest.main()
