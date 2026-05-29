from __future__ import annotations

import unittest

from src.sub_agents.company_researcher import COMPANY_RESEARCHER_SYSTEM_PROMPT


class CompanyResearcherPromptTests(unittest.TestCase):
    def test_company_researcher_does_not_request_unused_state_json(self) -> None:
        self.assertNotIn("Company State JSON", COMPANY_RESEARCHER_SYSTEM_PROMPT)
        self.assertNotIn('"companies": []', COMPANY_RESEARCHER_SYSTEM_PROMPT)
        self.assertIn("## Companies Researched", COMPANY_RESEARCHER_SYSTEM_PROMPT)
        self.assertIn("Preserve job URLs", COMPANY_RESEARCHER_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
