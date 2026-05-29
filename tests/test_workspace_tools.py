from __future__ import annotations

import json
import os
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

from src import task_runner
from src.codex_oauth.auth import PROJECT_ROOT, default_codex_auth_path
from src.configs import CONFIG
from src.tools import workspace_tools


class WorkspaceToolsTests(unittest.TestCase):
    def test_resolve_workspace_virtual_path(self) -> None:
        expected = CONFIG.workspace.latest_dir / "test.txt"

        self.assertEqual(workspace_tools._resolve_workspace_path("/latest/test.txt"), expected)
        self.assertEqual(workspace_tools._resolve_workspace_path("/workspace/latest/test.txt"), expected)
        self.assertEqual(workspace_tools._resolve_workspace_path("workspace/latest/test.txt"), expected)

    def test_start_workspace_run_clears_stale_latest_artifacts_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            latest_dir = root / "latest"
            runs_dir = root / "runs"
            latest_dir.mkdir(parents=True)
            stale_report = latest_dir / "06_final_job_search_report.md"
            stale_report.write_text("old report", encoding="utf-8")
            state_file = latest_dir / "job_search_state.json"
            state_file.write_text('{"schema_version": 1, "jobs": []}', encoding="utf-8")
            user_file = latest_dir / "notes.md"
            user_file.write_text("keep me", encoding="utf-8")

            with (
                patch.object(workspace_tools, "LATEST_DIR", latest_dir),
                patch.object(workspace_tools, "RUNS_DIR", runs_dir),
            ):
                result = workspace_tools.start_workspace_run.invoke({})

            self.assertTrue(result["success"], result)
            self.assertEqual(result["cleared_latest_artifacts"], ["06_final_job_search_report.md"])
            self.assertFalse(stale_report.exists())
            self.assertTrue(state_file.exists())
            self.assertEqual(user_file.read_text(encoding="utf-8"), "keep me")

    def test_save_job_artifact_rejects_invalid_run_id(self) -> None:
        result = workspace_tools.save_job_artifact.invoke(
            {
                "run_id": "../outside",
                "file_name": "01_intake_brief.md",
                "content": "brief",
            }
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "Invalid run_id. Use the run_id returned by start_workspace_run.")

    def test_update_job_search_state_rejects_invalid_run_id(self) -> None:
        result = workspace_tools.update_job_search_state.invoke(
            {
                "run_id": "../outside",
                "state_patch_json": "{}",
            }
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "Invalid run_id. Use the run_id returned by start_workspace_run.")

    def test_update_job_search_state_requires_existing_run_dir(self) -> None:
        result = workspace_tools.update_job_search_state.invoke(
            {
                "run_id": "2099-01-01_000000",
                "state_patch_json": "{}",
            }
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "Run directory does not exist. Call start_workspace_run first.")

    def test_update_job_search_state_accepts_existing_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            latest_dir = root / "latest"
            runs_dir = root / "runs"
            run_id = "2026-05-29_120000"
            (runs_dir / run_id).mkdir(parents=True)

            with (
                patch.object(workspace_tools, "LATEST_DIR", latest_dir),
                patch.object(workspace_tools, "RUNS_DIR", runs_dir),
            ):
                result = workspace_tools.update_job_search_state.invoke(
                    {
                        "run_id": run_id,
                        "state_patch_json": (
                            '{"jobs": [{"title": "Backend Engineer", "company": "Acme", '
                            '"location": "Leipzig", "canonical_url": "https://acme.example/jobs/1", '
                            '"requirements": ["Python"], "match_score": 88, "recommendation": "Apply"}], '
                            '"companies": [{"name": "Acme"}], "run": {"completed_at": "2026-05-29"}}'
                        ),
                    }
                )
                state = json.loads((latest_dir / "job_search_state.json").read_text(encoding="utf-8"))

        self.assertTrue(result["success"], result)
        self.assertEqual(result["job_count"], 1)
        self.assertEqual(list(state), ["jobs"])
        self.assertEqual(
            set(state["jobs"][0]),
            {
                "title",
                "company",
                "location",
                "canonical_url",
                "source_urls",
                "dedupe_key",
                "first_seen_at",
                "last_seen_at",
            },
        )
        self.assertNotIn("requirements", state["jobs"][0])
        self.assertNotIn("match_score", state["jobs"][0])
        self.assertNotIn("recommendation", state["jobs"][0])

    def test_update_job_search_state_from_artifact_extracts_named_json_section(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            latest_dir = root / "latest"
            runs_dir = root / "runs"
            run_id = "2026-05-29_120000"
            run_dir = runs_dir / run_id
            run_dir.mkdir(parents=True)
            artifact = run_dir / "02_raw_job_results.md"
            artifact.write_text(
                """
## Search Strategy
```json
{"not_state": true}
```

## Job State JSON
```json
{"jobs": [{"title": "Backend Engineer", "company": "Acme", "location": "Leipzig", "canonical_url": "https://acme.example/jobs/1"}]}
```
""".strip(),
                encoding="utf-8",
            )

            with (
                patch.object(workspace_tools, "LATEST_DIR", latest_dir),
                patch.object(workspace_tools, "RUNS_DIR", runs_dir),
            ):
                result = workspace_tools.update_job_search_state_from_artifact.invoke(
                    {
                        "run_id": run_id,
                        "file_name": "02_raw_job_results.md",
                        "state_section_heading": "Job State JSON",
                    }
                )
                state = json.loads((latest_dir / "job_search_state.json").read_text(encoding="utf-8"))

        self.assertTrue(result["success"], result)
        self.assertEqual(result["merged_job_count"], 1)
        self.assertEqual(state["jobs"][0]["canonical_url"], "https://acme.example/jobs/1")

    def test_update_job_search_state_from_artifact_requires_named_section(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runs_dir = root / "runs"
            run_id = "2026-05-29_120000"
            run_dir = runs_dir / run_id
            run_dir.mkdir(parents=True)
            (run_dir / "02_raw_job_results.md").write_text("## Other\n{}", encoding="utf-8")

            with patch.object(workspace_tools, "RUNS_DIR", runs_dir):
                result = workspace_tools.update_job_search_state_from_artifact.invoke(
                    {
                        "run_id": run_id,
                        "file_name": "02_raw_job_results.md",
                        "state_section_heading": "Job State JSON",
                    }
                )

        self.assertFalse(result["success"])
        self.assertIn("was not found", result["error"])

    def test_save_final_report_repairs_missing_url_cells_from_source_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            latest_dir = root / "latest"
            runs_dir = root / "runs"
            run_id = "2026-05-29_120000"
            run_dir = runs_dir / run_id
            run_dir.mkdir(parents=True)
            (run_dir / "03_verified_job_results.md").write_text(
                """
## Verified Job State JSON
```json
{"jobs": [
  {"title": "AI Developer (m/w/d) - Munich", "company": "Optimus Search", "verification_status": "unverified", "canonical_url": "https://jobs.example/optimus-ai-developer"},
  {"title": "Data Scientist (m/f/d) - remote", "company": "Unspecified", "verification_status": "closed", "canonical_url": "https://jobs.example/data-scientist-remote"},
  {"title": "Data Scientist (m|f|d)", "company": "MEAG", "verification_status": "closed", "canonical_url": "https://jobs.example/meag-data-scientist"},
  {"title": "AI Engineer", "company": "Knowunity GmbH", "verification_status": "access_blocked", "final_url": "https://jobs.example/knowunity-ai-engineer"}
]}
```
""".strip(),
                encoding="utf-8",
            )
            (run_dir / "02_raw_job_results.md").write_text(
                """
## Backlog Jobs
- Job: Senior C++ QT/ML Engineer — Infosys Ltd. / in-tech — München hybrid — https://jobs.example/senior-qt-ml-engineer — ML engineering signal.
- Job: Working Student in (Gen)AI Transformation Management (m|f|d) — MEAG — München — https://jobs.example/meag-working-student — student role.

## Job State JSON
```json
{"jobs": [
  {"title": "AWS AI & Data Engineer (m/w/d)", "company": "Reply / Storm Reply", "canonical_url": "https://jobs.example/aws-ai-data-engineer"}
]}
```
""".strip(),
                encoding="utf-8",
            )
            final_report = """
## Good But Unverified

| Role | Company | Verification status | Recommendation | URL |
| --- | --- | --- | --- | --- |
| AI Developer | Optimus Search | unverified | Unspecified | URL not supplied |

## Backlog To Verify Later

| Role | Company | Verification status | Recommendation | URL |
| --- | --- | --- | --- | --- |
| AWS AI & Data Engineer | Reply | not_verified_backlog | Unspecified | URL not supplied |
| Senior QT/ML Engineer | Infosys / in-tech | not_verified_backlog | Unspecified | URL not supplied |
| MEAG working student role, exact title not supplied | MEAG | not_verified_backlog | Unspecified | URL not supplied |

## Closed Or Access-Limited

| Role | Company | Verification status | Recommendation | URL | Notes |
| --- | --- | --- | --- | --- | --- |
| Data Scientist, remote listing with company not supplied | Unspecified | closed | Skip | URL not supplied | Posting appeared unavailable. |
| AI Engineer | Knowunity | access_blocked | Unspecified | URL not supplied | Crawler could not read the page. |
""".strip()

            with (
                patch.object(workspace_tools, "LATEST_DIR", latest_dir),
                patch.object(workspace_tools, "RUNS_DIR", runs_dir),
            ):
                result = workspace_tools.save_job_artifact.invoke(
                    {
                        "run_id": run_id,
                        "file_name": "06_final_job_search_report.md",
                        "content": final_report,
                    }
                )
                saved_report = (latest_dir / "06_final_job_search_report.md").read_text(encoding="utf-8")
                run_report = (run_dir / "06_final_job_search_report.md").read_text(encoding="utf-8")

        self.assertTrue(result["success"], result)
        self.assertEqual(saved_report, run_report)
        self.assertNotIn("URL not supplied", saved_report)
        self.assertIn("[Job page](https://jobs.example/optimus-ai-developer)", saved_report)
        self.assertIn("[Job page](https://jobs.example/aws-ai-data-engineer)", saved_report)
        self.assertIn("[Job page](https://jobs.example/senior-qt-ml-engineer)", saved_report)
        self.assertIn("[Job page](https://jobs.example/meag-working-student)", saved_report)
        self.assertIn("[Job page](https://jobs.example/data-scientist-remote)", saved_report)
        self.assertNotIn("[Job page](https://jobs.example/meag-data-scientist)", saved_report)
        self.assertIn("[Job page](https://jobs.example/knowunity-ai-engineer)", saved_report)

    def test_read_job_search_dedupe_state_returns_compact_brief(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            latest_dir = Path(temp_dir) / "latest"
            latest_dir.mkdir(parents=True)
            (latest_dir / "job_search_state.json").write_text(
                json.dumps(
                    {
                        "jobs": [
                            {
                                "title": "Backend Engineer",
                                "company": "Acme",
                                "location": "Leipzig",
                                "canonical_url": "https://acme.example/jobs/1",
                                "dedupe_key": "acme|backend engineer|leipzig",
                                "status": "active",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(workspace_tools, "LATEST_DIR", latest_dir):
                result = workspace_tools.read_job_search_dedupe_state.invoke({"limit": 10})

        self.assertTrue(result["success"], result)
        self.assertEqual(result["job_count"], 1)
        self.assertEqual(result["returned_job_count"], 1)
        self.assertIn("acme|backend_engineer|leipzig", result["dedupe_brief"])
        self.assertIn("status=active_or_previous", result["dedupe_brief"])
        self.assertIn("last_seen=", result["dedupe_brief"])
        self.assertEqual(result["jobs"][0]["canonical_url"], "https://acme.example/jobs/1")

    def test_read_job_search_dedupe_state_prioritizes_current_recent_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            latest_dir = Path(temp_dir) / "latest"
            latest_dir.mkdir(parents=True)
            (latest_dir / "job_search_state.json").write_text(
                json.dumps(
                    {
                        "jobs": [
                            {
                                "title": "Active stale",
                                "company": "Acme",
                                "location": "Leipzig",
                                "canonical_url": "https://acme.example/jobs/active-stale",
                                "last_seen_at": "2026-05-10",
                            },
                            {
                                "title": "Closed newest",
                                "company": "Acme",
                                "location": "Leipzig",
                                "canonical_url": "https://acme.example/jobs/closed-newest",
                                "status": "closed",
                                "last_seen_at": "2026-05-30",
                            },
                            {
                                "title": "Unknown recent",
                                "company": "Acme",
                                "location": "Leipzig",
                                "canonical_url": "https://acme.example/jobs/unknown-recent",
                                "status": "unknown",
                                "last_seen_at": "2026-05-28",
                            },
                            {
                                "title": "Active recent",
                                "company": "Acme",
                                "location": "Leipzig",
                                "canonical_url": "https://acme.example/jobs/active-recent",
                                "last_seen_at": "2026-05-29",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(workspace_tools, "LATEST_DIR", latest_dir):
                result = workspace_tools.read_job_search_dedupe_state.invoke({"limit": 3})

        self.assertEqual(
            [job["title"] for job in result["jobs"]],
            ["Active recent", "Unknown recent", "Active stale"],
        )
        self.assertNotIn("Closed newest", result["dedupe_brief"])

    def test_loads_jsonish_prefers_explicit_json_fenced_block(self) -> None:
        payload = """
```python
print("not json")
```

```json
{"jobs": []}
```
"""

        self.assertEqual(workspace_tools._loads_jsonish(payload), {"jobs": []})

    def test_merge_job_patch_keeps_only_dedupe_details(self) -> None:
        now = "2026-05-28T00:00:00+02:00"
        existing = [
            {
                "title": "Backend Engineer",
                "company": "Acme",
                "location": "Leipzig",
                "canonical_url": "https://acme.com/jobs/backend",
                "requirements": ["Python", "Docker"],
                "confidence": "High",
                "source_urls": ["https://acme.com/jobs/backend"],
            }
        ]
        match_patch = [
            {
                "title": "Backend Engineer",
                "company": "Acme",
                "location": "Leipzig",
                "canonical_url": "https://acme.com/jobs/backend",
                "match_score": 80,
                "recommendation": "Apply",
            }
        ]

        merged = workspace_tools._merge_jobs(existing, match_patch, now)

        self.assertEqual(merged[0]["title"], "Backend Engineer")
        self.assertEqual(merged[0]["company"], "Acme")
        self.assertEqual(merged[0]["location"], "Leipzig")
        self.assertEqual(merged[0]["canonical_url"], "https://acme.com/jobs/backend")
        self.assertNotIn("status", merged[0])
        self.assertNotIn("requirements", merged[0])
        self.assertNotIn("confidence", merged[0])
        self.assertNotIn("match_score", merged[0])
        self.assertNotIn("recommendation", merged[0])

    def test_url_normalization_merges_tracking_variants(self) -> None:
        now = "2026-05-28T00:00:00+02:00"
        jobs = [
            {
                "title": "Senior Backend Engineer",
                "company": "Acme GmbH",
                "location": "München",
                "canonical_url": "https://EXAMPLE.com/jobs/1?utm_source=linkedin#apply",
            },
            {
                "title": "Backend Engineer",
                "company": "Acme",
                "location": "Munich",
                "canonical_url": "https://example.com/jobs/1",
            },
        ]

        merged = workspace_tools._merge_jobs([], jobs, now)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["canonical_url"], "https://example.com/jobs/1")
        self.assertEqual(merged[0]["source_urls"], ["https://example.com/jobs/1"])

    def test_dedupe_key_is_normalized_from_job_fields(self) -> None:
        now = "2026-05-28T00:00:00+02:00"
        jobs = [
            {
                "title": "Backend Engineer",
                "company": "Acme",
                "location": "München",
                "canonical_url": "https://acme.example/jobs/1",
                "dedupe_key": "Acme | Backend Engineer | München",
            },
            {
                "title": "Backend Engineer",
                "company": "Acme",
                "location": "München",
                "canonical_url": "https://acme.example/jobs/2",
            },
        ]

        merged = workspace_tools._merge_jobs([], jobs, now)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["dedupe_key"], "acme|backend_engineer|munchen")

    def test_missing_or_placeholder_location_does_not_merge_different_urls(self) -> None:
        now = "2026-05-28T00:00:00+02:00"
        jobs = [
            {
                "title": "Backend Engineer",
                "company": "Acme",
                "location": "Unspecified",
                "canonical_url": "https://acme.example/jobs/berlin",
            },
            {
                "title": "Backend Engineer",
                "company": "Acme",
                "location": "",
                "canonical_url": "https://acme.example/jobs/munich",
            },
        ]

        merged = workspace_tools._merge_jobs([], jobs, now)

        self.assertEqual(len(merged), 2)
        self.assertEqual(
            {job["canonical_url"] for job in merged},
            {"https://acme.example/jobs/berlin", "https://acme.example/jobs/munich"},
        )
        self.assertTrue(all("dedupe_key" not in job for job in merged))

    def test_placeholder_list_values_do_not_replace_existing_job_details(self) -> None:
        now = "2026-05-28T00:00:00+02:00"
        existing = [
            {
                "title": "Backend Engineer",
                "company": "Acme",
                "location": "Leipzig",
                "canonical_url": "https://acme.com/jobs/backend",
                "requirements": ["Python", "Docker"],
                "source_urls": ["https://acme.com/jobs/backend"],
            }
        ]
        placeholder_patch = workspace_tools._compact_state_patch(
            {
                "jobs": [
                    {
                        "title": "Backend Engineer",
                        "company": "Acme",
                        "location": "Leipzig",
                        "canonical_url": "https://acme.com/jobs/backend",
                        "requirements": "Unspecified",
                    }
                ]
            }
        )

        merged = workspace_tools._merge_jobs(existing, placeholder_patch["jobs"], now)

        self.assertEqual(merged[0]["canonical_url"], "https://acme.com/jobs/backend")
        self.assertNotIn("requirements", merged[0])

    def test_closed_verification_status_persists_compact_lifecycle_status(self) -> None:
        now = "2026-05-28T00:00:00+02:00"

        merged = workspace_tools._merge_jobs(
            [],
            [
                {
                    "title": "Backend Engineer",
                    "company": "Acme",
                    "location": "Leipzig",
                    "canonical_url": "https://acme.com/jobs/backend",
                    "verification_status": "closed",
                }
            ],
            now,
        )

        self.assertEqual(merged[0]["canonical_url"], "https://acme.com/jobs/backend")
        self.assertEqual(merged[0]["status"], "closed")
        self.assertNotIn("verification_status", merged[0])

    def test_closed_status_does_not_stick_to_newly_discovered_url(self) -> None:
        now = "2026-05-28T00:00:00+02:00"
        existing = [
            {
                "title": "Backend Engineer",
                "company": "Acme",
                "location": "Leipzig",
                "canonical_url": "https://acme.com/jobs/old-backend",
                "status": "closed",
            }
        ]
        newly_discovered = [
            {
                "title": "Backend Engineer",
                "company": "Acme",
                "location": "Leipzig",
                "canonical_url": "https://acme.com/jobs/new-backend",
            }
        ]

        merged = workspace_tools._merge_jobs(existing, newly_discovered, now)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["canonical_url"], "https://acme.com/jobs/new-backend")
        self.assertEqual(
            merged[0]["source_urls"],
            ["https://acme.com/jobs/old-backend", "https://acme.com/jobs/new-backend"],
        )
        self.assertNotIn("status", merged[0])

    def test_verified_patch_keeps_only_dedupe_fields(self) -> None:
        now = "2026-05-28T00:00:00+02:00"
        existing = [
            {
                "title": "Backend Engineer",
                "company": "Acme",
                "location": "Leipzig",
                "canonical_url": "https://acme.com/jobs/backend",
                "verification_status": "closed",
                "status": "closed",
            }
        ]
        verified_patch = [
            {
                "title": "Backend Engineer",
                "company": "Acme",
                "location": "Leipzig",
                "canonical_url": "https://acme.com/jobs/backend",
                "verification_status": "verified",
            }
        ]

        merged = workspace_tools._merge_jobs(existing, verified_patch, now)

        self.assertEqual(merged[0]["canonical_url"], "https://acme.com/jobs/backend")
        self.assertNotIn("status", merged[0])
        self.assertNotIn("verification_status", merged[0])

    def test_sparse_match_patch_keeps_only_dedupe_fields(self) -> None:
        now = "2026-05-28T00:00:00+02:00"
        existing = [
            {
                "title": "Backend Engineer",
                "company": "Acme",
                "location": "Leipzig",
                "canonical_url": "https://acme.com/jobs/backend",
                "verification_status": "closed",
                "status": "closed",
            }
        ]
        match_patch = [
            {
                "title": "Backend Engineer",
                "company": "Acme",
                "location": "Leipzig",
                "canonical_url": "https://acme.com/jobs/backend",
                "match_score": 83,
                "recommendation": "Apply",
            }
        ]

        merged = workspace_tools._merge_jobs(existing, match_patch, now)

        self.assertEqual(merged[0]["canonical_url"], "https://acme.com/jobs/backend")
        self.assertEqual(merged[0]["status"], "closed")
        self.assertNotIn("verification_status", merged[0])
        self.assertNotIn("match_score", merged[0])

    def test_uncertain_verification_status_persists_unknown_lifecycle_status(self) -> None:
        now = "2026-05-28T00:00:00+02:00"
        existing = [
            {
                "title": "Backend Engineer",
                "company": "Acme",
                "location": "Leipzig",
                "canonical_url": "https://acme.com/jobs/backend",
                "verification_status": "closed",
                "status": "closed",
            }
        ]

        for verification_status in (
            "access_blocked",
            "login_required",
            "not_verified_backlog",
            "unverified",
        ):
            with self.subTest(verification_status=verification_status):
                merged = workspace_tools._merge_jobs(
                    existing,
                    [
                        {
                            "title": "Backend Engineer",
                            "company": "Acme",
                            "location": "Leipzig",
                            "canonical_url": "https://acme.com/jobs/backend",
                            "verification_status": verification_status,
                        }
                    ],
                    now,
                )

                self.assertEqual(merged[0]["canonical_url"], "https://acme.com/jobs/backend")
                self.assertEqual(merged[0]["status"], "closed")
                self.assertNotIn("verification_status", merged[0])

    def test_new_uncertain_verification_status_keeps_only_dedupe_fields(self) -> None:
        now = "2026-05-28T00:00:00+02:00"

        merged = workspace_tools._merge_jobs(
            [],
            [
                {
                    "title": "Backend Engineer",
                    "company": "Acme",
                    "location": "Leipzig",
                    "canonical_url": "https://acme.com/jobs/backend",
                    "verification_status": "access_blocked",
                }
            ],
            now,
        )

        self.assertEqual(merged[0]["canonical_url"], "https://acme.com/jobs/backend")
        self.assertEqual(merged[0]["status"], "unknown")
        self.assertNotIn("verification_status", merged[0])

    def test_uncertain_verification_status_marks_existing_active_as_unknown(self) -> None:
        now = "2026-05-28T00:00:00+02:00"
        existing = [
            {
                "title": "Backend Engineer",
                "company": "Acme",
                "location": "Leipzig",
                "canonical_url": "https://acme.com/jobs/backend",
                "status": "active",
            }
        ]

        merged = workspace_tools._merge_jobs(
            existing,
            [
                {
                    "title": "Backend Engineer",
                    "company": "Acme",
                    "location": "Leipzig",
                    "canonical_url": "https://acme.com/jobs/backend",
                    "verification_status": "access_blocked",
                }
            ],
            now,
        )

        self.assertEqual(merged[0]["canonical_url"], "https://acme.com/jobs/backend")
        self.assertEqual(merged[0]["status"], "unknown")
        self.assertNotIn("verification_status", merged[0])

    def test_merge_company_patch_preserves_existing_research_details(self) -> None:
        now = "2026-05-28T00:00:00+02:00"
        existing = [
            {
                "name": "Acme GmbH",
                "website": "https://acme.example",
                "industry": "Fintech",
                "locations": ["Leipzig"],
                "research_status": "researched",
                "last_researched_at": "2026-05-01T00:00:00+02:00",
                "summary": "Useful existing research",
                "risks": ["Visa status unclear"],
                "interview_prep": ["Ask about platform scale"],
            }
        ]
        sparse_patch = [
            {
                "name": "Acme GmbH",
                "website": "Unspecified",
                "industry": "",
                "locations": [],
                "research_status": "not_started",
                "last_researched_at": "",
                "summary": "Unspecified",
                "risks": [],
                "interview_prep": [],
            }
        ]

        merged = workspace_tools._merge_companies(existing, sparse_patch, now)

        self.assertEqual(merged[0]["website"], "https://acme.example")
        self.assertEqual(merged[0]["industry"], "Fintech")
        self.assertEqual(merged[0]["locations"], ["Leipzig"])
        self.assertEqual(merged[0]["research_status"], "researched")
        self.assertEqual(merged[0]["last_researched_at"], "2026-05-01T00:00:00+02:00")
        self.assertEqual(merged[0]["summary"], "Useful existing research")
        self.assertEqual(merged[0]["risks"], ["Visa status unclear"])
        self.assertEqual(merged[0]["interview_prep"], ["Ask about platform scale"])

    def test_placeholder_list_values_do_not_replace_existing_company_details(self) -> None:
        now = "2026-05-28T00:00:00+02:00"
        existing = [
            {
                "name": "Acme GmbH",
                "research_status": "researched",
                "risks": ["Visa status unclear"],
                "interview_prep": ["Ask about platform scale"],
            }
        ]
        placeholder_patch = {
            "companies": [
                workspace_tools._compact_company_for_state(
                    {
                        "name": "Acme GmbH",
                        "risks": "Unknown",
                        "interview_prep": "Unspecified",
                    }
                )
            ]
        }

        merged = workspace_tools._merge_companies(existing, placeholder_patch["companies"], now)

        self.assertEqual(merged[0]["risks"], ["Visa status unclear"])
        self.assertEqual(merged[0]["interview_prep"], ["Ask about platform scale"])


class CodexAuthPathTests(unittest.TestCase):
    def test_default_codex_auth_path_uses_project_root(self) -> None:
        with patch.dict(os.environ, {"JOB_FINDER_CODEX_AUTH_PATH": "", "CODEX_OAUTH_AUTH_PATH": ""}):
            self.assertEqual(
                default_codex_auth_path(),
                PROJECT_ROOT / ".codex_oauth" / "auth" / "codex.json",
            )

    def test_relative_codex_auth_override_uses_project_root(self) -> None:
        with patch.dict(os.environ, {"JOB_FINDER_CODEX_AUTH_PATH": "local/auth.json", "CODEX_OAUTH_AUTH_PATH": ""}):
            self.assertEqual(default_codex_auth_path(), PROJECT_ROOT / "local" / "auth.json")


class JobSearchTaskTests(unittest.TestCase):
    def test_build_task_injects_search_date_when_missing(self) -> None:
        task = task_runner.build_task(
            {"target_roles": ["Backend Engineer"]},
            search_date="2026-05-29",
        )

        self.assertTrue(task.startswith("search_date: 2026-05-29\n"))
        self.assertIn("target_roles: Backend Engineer", task)

    def test_build_task_preserves_explicit_search_date(self) -> None:
        task = task_runner.build_task(
            {
                "search_date": "2026-05-28",
                "target_roles": ["Backend Engineer"],
            },
            search_date="2026-05-29",
        )

        self.assertEqual(task.count("search_date:"), 1)
        self.assertIn("search_date: 2026-05-28", task)

    def test_default_job_search_task_uses_config_profile(self) -> None:
        self.assertEqual(
            task_runner.resolve_task_config_path(),
            PROJECT_ROOT / "src" / "configs" / "job_search.py",
        )

        task = task_runner.load_job_search_task()

        self.assertIn("target_roles: ", task)
        self.assertIn("target_locations: München", task)
        self.assertIn("candidate_skills: python, AI", task)
        self.assertIn("resume_pdf_path: ", task)

    def test_load_job_search_task_from_python(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "job_search.py"
            config_path.write_text(
                """
job_search = {
    "target_roles": ["Backend Engineer", "Platform Engineer"],
    "target_locations": ["Leipzig"],
    "remote_preference": "hybrid or remote",
    "notes": "focus on Python\\nand Docker",
}
""".strip(),
                encoding="utf-8",
            )

            task = task_runner.load_job_search_task(config_path)

        self.assertIn("target_roles: Backend Engineer, Platform Engineer", task)
        self.assertIn("target_locations: Leipzig", task)
        self.assertIn("remote_preference: hybrid or remote", task)
        self.assertIn("notes: focus on Python and Docker", task)
        self.assertIn("resume_pdf_path: ", task)

    def test_load_job_search_task_rejects_toml(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "job_search.toml"
            config_path.write_text(
                """
[job_search]
target_roles = ["Backend Engineer", "Platform Engineer"]
target_locations = ["Leipzig"]
remote_preference = "hybrid or remote"
notes = "focus on Python\\nand Docker"
""".strip(),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                task_runner.load_job_search_task(config_path)

    def test_resolve_task_config_path_prefers_env_var(self) -> None:
        with patch.dict(os.environ, {task_runner.TASK_CONFIG_ENV_VAR: "src/configs/custom.py"}):
            self.assertEqual(
                task_runner.resolve_task_config_path(),
                (PROJECT_ROOT / "src" / "configs" / "custom.py").resolve(),
            )


class PyprojectPackagingTests(unittest.TestCase):
    def test_wheel_target_preserves_src_runtime_package(self) -> None:
        pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

        wheel_target = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]

        self.assertEqual(wheel_target["only-include"], ["src"])
        self.assertNotIn("packages", wheel_target)


if __name__ == "__main__":
    unittest.main()
