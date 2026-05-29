from __future__ import annotations

import unittest

from src.sub_agents.resume_matcher import RESUME_MATCHER_SYSTEM_PROMPT


class ResumeMatcherPromptTests(unittest.TestCase):
    def test_no_resume_max_score_matches_rubric_caps(self) -> None:
        self.assertIn("The maximum match_score is 71.", RESUME_MATCHER_SYSTEM_PROMPT)
        self.assertNotIn("The maximum match_score is 80.", RESUME_MATCHER_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
