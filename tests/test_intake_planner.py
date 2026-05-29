from __future__ import annotations

import unittest

from src.sub_agents.intake_planner import (
    INTAKE_PLANNER_MIDDLEWARE,
    INTAKE_PLANNER_SYSTEM_PROMPT,
    detect_phone_numbers,
)


class IntakePlannerMiddlewareTests(unittest.TestCase):
    def test_intake_planner_does_not_redact_job_relevant_urls(self) -> None:
        pii_types = {
            middleware.pii_type
            for middleware in INTAKE_PLANNER_MIDDLEWARE
            if hasattr(middleware, "pii_type")
        }

        self.assertIn("email", pii_types)
        self.assertIn("phone", pii_types)
        self.assertNotIn("url", pii_types)

    def test_intake_planner_preserves_search_date_context(self) -> None:
        self.assertIn("- Search date:", INTAKE_PLANNER_SYSTEM_PROMPT)
        self.assertIn("Preserve any user-provided search_date", INTAKE_PLANNER_SYSTEM_PROMPT)

    def test_phone_detector_preserves_dates_and_timelines(self) -> None:
        text = "Search date 2026-05-28. Start date 2026-06. Experience from 2021 05 28."

        self.assertEqual(detect_phone_numbers(text), [])

    def test_phone_detector_preserves_salary_ranges_and_year_series(self) -> None:
        text = (
            "Salary 60.000-80.000 EUR, alternative 60 000 80 000 EUR. "
            "Experience years 2020 2021 2022."
        )

        self.assertEqual(detect_phone_numbers(text), [])

    def test_phone_detector_still_finds_real_phone_numbers(self) -> None:
        matches = detect_phone_numbers("Candidate phone +49 151 12345678.")

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["value"], "+49 151 12345678")

    def test_phone_detector_still_finds_local_phone_numbers(self) -> None:
        matches = detect_phone_numbers("Candidate phone 030 123 4567.")

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["value"], "030 123 4567")


if __name__ == "__main__":
    unittest.main()
