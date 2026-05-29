from __future__ import annotations

import unittest

from src.configs import CONFIG
from src.sub_agents.job_searcher import JOB_SEARCHER_SYSTEM_PROMPT


class JobSearcherPromptTests(unittest.TestCase):
    def test_enterprise_direct_results_are_capped_in_ranked_results(self) -> None:
        self.assertIn(
            f"At most 5 of the top {CONFIG.search.max_detailed_jobs} detailed results may be enterprise/direct",
            JOB_SEARCHER_SYSTEM_PROMPT,
        )
        self.assertIn("companies in or near the target locations", JOB_SEARCHER_SYSTEM_PROMPT)
        self.assertIn("previous job dedupe brief", JOB_SEARCHER_SYSTEM_PROMPT)
        self.assertIn("already-seen jobs", JOB_SEARCHER_SYSTEM_PROMPT)
        self.assertIn("active or unknown-status", JOB_SEARCHER_SYSTEM_PROMPT)
        self.assertIn("status=closed", JOB_SEARCHER_SYSTEM_PROMPT)
        self.assertIn("lowercase ASCII, accent folding", JOB_SEARCHER_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
