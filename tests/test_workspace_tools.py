from __future__ import annotations

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
                        "state_patch_json": '{"jobs": [{"title": "Backend Engineer", "company": "Acme"}]}',
                    }
                )

        self.assertTrue(result["success"], result)
        self.assertEqual(result["job_count"], 1)

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

    def test_merge_job_patch_preserves_existing_details(self) -> None:
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

        self.assertEqual(merged[0]["requirements"], ["Python", "Docker"])
        self.assertEqual(merged[0]["confidence"], "High")
        self.assertEqual(merged[0]["match_score"], 80)
        self.assertEqual(merged[0]["recommendation"], "Apply")

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

        self.assertEqual(merged[0]["requirements"], ["Python", "Docker"])

    def test_closed_verification_status_sets_lifecycle_status_closed(self) -> None:
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

        self.assertEqual(merged[0]["status"], "closed")
        self.assertEqual(merged[0]["verification_status"], "closed")

    def test_verified_patch_reopens_previous_closed_status(self) -> None:
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

        self.assertEqual(merged[0]["verification_status"], "verified")
        self.assertEqual(merged[0]["status"], "active")

    def test_sparse_match_patch_does_not_reopen_closed_status(self) -> None:
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

        self.assertEqual(merged[0]["verification_status"], "closed")
        self.assertEqual(merged[0]["status"], "closed")
        self.assertEqual(merged[0]["match_score"], 83)

    def test_uncertain_verification_status_does_not_reopen_closed_status(self) -> None:
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

                self.assertEqual(merged[0]["status"], "closed")
                self.assertEqual(merged[0]["verification_status"], verification_status)

    def test_new_uncertain_verification_status_has_unknown_lifecycle_status(self) -> None:
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

        self.assertEqual(merged[0]["status"], "unknown")
        self.assertEqual(merged[0]["verification_status"], "access_blocked")

    def test_uncertain_verification_status_updates_active_lifecycle_to_unknown(self) -> None:
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

        self.assertEqual(merged[0]["status"], "unknown")
        self.assertEqual(merged[0]["verification_status"], "access_blocked")

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
        placeholder_patch = workspace_tools._compact_state_patch(
            {
                "companies": [
                    {
                        "name": "Acme GmbH",
                        "risks": "Unknown",
                        "interview_prep": "Unspecified",
                    }
                ]
            }
        )

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
            PROJECT_ROOT / "src" / "configs" / "job_search.toml",
        )

        task = task_runner.load_job_search_task()

        self.assertIn("target_roles: Backend Engineer", task)
        self.assertIn("target_locations: Leipzig", task)
        self.assertIn("resume_pdf_path: ", task)

    def test_load_job_search_task_from_toml(self) -> None:
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

            task = task_runner.load_job_search_task(config_path)

        self.assertIn("target_roles: Backend Engineer, Platform Engineer", task)
        self.assertIn("target_locations: Leipzig", task)
        self.assertIn("remote_preference: hybrid or remote", task)
        self.assertIn("notes: focus on Python and Docker", task)
        self.assertIn("resume_pdf_path: ", task)

    def test_resolve_task_config_path_prefers_env_var(self) -> None:
        with patch.dict(os.environ, {task_runner.TASK_CONFIG_ENV_VAR: "src/configs/custom.toml"}):
            self.assertEqual(
                task_runner.resolve_task_config_path(),
                (PROJECT_ROOT / "src" / "configs" / "custom.toml").resolve(),
            )


class PyprojectPackagingTests(unittest.TestCase):
    def test_wheel_target_preserves_src_runtime_package(self) -> None:
        pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

        wheel_target = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]

        self.assertEqual(wheel_target["only-include"], ["src"])
        self.assertNotIn("packages", wheel_target)


if __name__ == "__main__":
    unittest.main()
