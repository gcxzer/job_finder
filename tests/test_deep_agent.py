from __future__ import annotations

import unittest
from unittest.mock import patch

from src import deep_agent


class LazyDeepAgentTests(unittest.TestCase):
    def test_lazy_deep_agent_does_not_build_until_used(self) -> None:
        class FakeGraph:
            def invoke(self, *args: object, **kwargs: object) -> tuple[str, tuple[object, ...], dict[str, object]]:
                return "invoke", args, kwargs

        lazy_agent = deep_agent.LazyDeepAgent()

        with patch.object(deep_agent, "build_deep_agent", return_value=FakeGraph()) as build:
            self.assertFalse(lazy_agent.is_loaded)
            build.assert_not_called()

            result = lazy_agent.invoke("payload", config={"thread_id": "test"})

        self.assertTrue(lazy_agent.is_loaded)
        build.assert_called_once_with()
        self.assertEqual(result, ("invoke", ("payload",), {"config": {"thread_id": "test"}}))


class MainAgentPromptTests(unittest.TestCase):
    def test_main_agent_reads_dedupe_state_before_search(self) -> None:
        self.assertIn("Call read_job_search_dedupe_state", deep_agent.MAIN_AGENT_SYSTEM_PROMPT)
        self.assertIn("Include the dedupe_brief", deep_agent.MAIN_AGENT_SYSTEM_PROMPT)
        self.assertIn("workspace job dedupe index", deep_agent.MAIN_AGENT_SYSTEM_PROMPT)
        self.assertNotIn("Company State JSON from the output.\n   - Call update_job_search_state", deep_agent.MAIN_AGENT_SYSTEM_PROMPT)

    def test_main_agent_uses_artifact_state_update_tool(self) -> None:
        self.assertIn("update_job_search_state_from_artifact", deep_agent.MAIN_AGENT_SYSTEM_PROMPT)
        self.assertIn('file_name="02_raw_job_results.md"', deep_agent.MAIN_AGENT_SYSTEM_PROMPT)
        self.assertIn('state_section_heading="Job State JSON"', deep_agent.MAIN_AGENT_SYSTEM_PROMPT)
        self.assertIn('file_name="03_verified_job_results.md"', deep_agent.MAIN_AGENT_SYSTEM_PROMPT)
        self.assertIn('state_section_heading="Verified Job State JSON"', deep_agent.MAIN_AGENT_SYSTEM_PROMPT)
        self.assertIn("retry the state", deep_agent.MAIN_AGENT_SYSTEM_PROMPT)
        self.assertIn("missing or contains invalid JSON", deep_agent.MAIN_AGENT_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
