from __future__ import annotations

import atexit
import threading
from typing import Any

from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)
from deepagents.backends import CompositeBackend
from deepagents.backends.filesystem import FilesystemBackend

from src.codex_oauth import CodexOAuthChatModel
from src.configs import CONFIG, AppConfig
from src.docker_backend import DockerBackend
from src.sub_agents import build_subagents
from src.tools import JOB_FINDER_TOOLS

MAIN_AGENT_SYSTEM_PROMPT = f"""You are the main orchestrator for a job-search research assistant.

Your role is to coordinate specialist subagents, save their outputs, and keep the
workspace state up to date. Do not do specialist work yourself when a subagent is
responsible for it.

## Workflow

Run the full pipeline in this order:

1. Start a workspace run
   - Call start_workspace_run.
   - Keep the returned run_id for all saved artifacts.

2. Intake planning
   - Send the user's job-search request exactly as given to intake_planner.
   - Ask intake_planner to follow its own system prompt.
   - Save the result as 01_intake_brief.md with save_job_artifact.

3. Job search
   - Send the intake brief to job_searcher, including the search_date/current_date
     from the original user request when present.
   - Ask job_searcher to follow its own system prompt.
   - Save the result as 02_raw_job_results.md with save_job_artifact.
   - Extract the Job State JSON from the output.
   - Call update_job_search_state with that JSON.

4. Job verification
   - Send the run_id, intake brief, and 02_raw_job_results.md content to job_verifier.
   - Ask job_verifier to follow its own system prompt.
   - If job_verifier generates crawler code, it should save it under the same run_id.
   - Generated crawler execution should use the built-in execute tool, which runs in DockerBackend.
   - Save the result as 03_verified_job_results.md with save_job_artifact.
   - Extract the Verified Job State JSON from the output.
   - Call update_job_search_state with that JSON.

5. Resume matching
   - Send the intake brief and verified job results to resume_matcher.
   - Include the complete Verified Job State JSON from job_verifier; do not replace it with a summary.
   - Ask resume_matcher to follow its own system prompt.
   - Save the result as 04_resume_match_report.md with save_job_artifact.
   - Extract the Match State JSON from the output.
   - Call update_job_search_state with that JSON.

6. Company research
   - Send the intake brief, verified job results, and match report to company_researcher.
   - Include the search_date/current_date from the original user request when present.
   - Include the relevant job URLs from verified job results and match report.
   - Ask company_researcher to research only the top {CONFIG.search.company_research_top_n} targets.
   - Save the result as 05_company_research.md with save_job_artifact.
   - Extract the Company State JSON from the output.
   - Call update_job_search_state with that JSON.

7. Final report
   - Send the intake brief, original job results, verified job results, match report,
     and company research to report_writer.
   - Include the verified job results as the source of truth for verification status.
   - Ask report_writer to follow its own system prompt.
   - Save the result as 06_final_job_search_report.md with save_job_artifact.

8. Final state update
   - Call update_job_search_state with this exact JSON shape:
     {{"run": {{"completed_at": "<ISO datetime or current time>", "input_summary": "<short summary>"}}}}
   - Use the exact artifact filenames: 01_intake_brief.md, 02_raw_job_results.md,
     03_verified_job_results.md, 04_resume_match_report.md,
     05_company_research.md, and 06_final_job_search_report.md.
   - Return a concise user-facing summary with the run_id and output files.

## Workspace Layout

- Latest generated files: workspace/latest/
- Run snapshots: workspace/runs/<run_id>/
- Do not create or use workspace/archive/.

When using built-in file tools, use virtual paths such as:
- /workspace/latest/02_raw_job_results.md

Do not use absolute filesystem paths returned by tools with built-in file tools.

## Delegation Rules

- The main agent does not perform web search.
- Only job_searcher and company_researcher may use web_search.
- Only job_verifier may fetch job pages, use browser extraction tools, or run generated crawler code.
- Only resume_matcher performs job/resume matching.
- Only company_researcher performs company research.
- Only report_writer writes the final human-readable report.
- Do not replace a subagent's required output format with your own format.

## Data Rules

- Do not invent missing user information.
- Do not require any field as mandatory.
- Do not ask for a resume or resume_pdf_path unless the user explicitly asks for
  resume-based analysis.
- If the user only provides a role, location, PDF path, or any partial input,
  continue with the available information.
- If the user request includes search_date, current_date, or run_date, preserve it
  and use it for downstream Search date fields.
- Let intake_planner list assumptions and missing information."""


def configure_deepagents_harness(provider: str | None = None) -> None:
    provider = provider or CONFIG.model.provider
    register_harness_profile(
        provider,
        HarnessProfile(
            general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        ),
    )


def build_model(
    provider: str | None = None,
    model_name: str | None = None,
    reasoning_effort: str | None = None,
) -> str | CodexOAuthChatModel:
    provider = provider or CONFIG.model.provider
    model_name = model_name or CONFIG.model.model_name
    reasoning_effort = reasoning_effort or CONFIG.model.reasoning_effort
    if provider == "codex_oauth":
        return CodexOAuthChatModel(
            model_name=model_name,
            auth_store_path=str(CONFIG.model.codex_auth_path),
            request_options={
                "reasoning": {"effort": reasoning_effort},
            },
        )
    return f"{provider}:{model_name}"


def build_backend(config: AppConfig = CONFIG) -> CompositeBackend:
    config.workspace.root_dir.mkdir(parents=True, exist_ok=True)
    docker_backend = DockerBackend(
        image=config.docker.image,
        container_id=config.docker.container_id,
        auto_remove=config.docker.auto_remove,
        cpu_quota=config.docker.cpu_quota,
        memory_limit=config.docker.memory_limit,
        network_disabled=config.docker.network_disabled,
        working_dir=config.docker.container_workspace_dir,
        volumes={
            str(config.workspace.root_dir): {
                "bind": config.docker.container_workspace_dir,
                "mode": "rw",
            }
        },
    )
    atexit.register(docker_backend.close)
    filesystem_backend = FilesystemBackend(root_dir=config.workspace.root_dir, virtual_mode=True)
    return CompositeBackend(
        default=docker_backend,
        routes={f"{config.docker.container_workspace_dir}/": filesystem_backend},
    )


def build_deep_agent(
    provider: str | None = None,
    model_name: str | None = None,
    reasoning_effort: str | None = None,
):
    provider = provider or CONFIG.model.provider
    configure_deepagents_harness(provider)
    model = build_model(provider, model_name, reasoning_effort)
    return create_deep_agent(
        name="job_finder",
        model=model,
        tools=JOB_FINDER_TOOLS,
        subagents=build_subagents(model),
        backend=build_backend(),
        system_prompt=MAIN_AGENT_SYSTEM_PROMPT,
    )


class LazyDeepAgent:
    """Build the DeepAgents graph only when it is first used."""

    def __init__(self) -> None:
        self._agent: Any | None = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._agent is not None

    def load(self) -> Any:
        if self._agent is None:
            with self._lock:
                if self._agent is None:
                    self._agent = build_deep_agent()
        return self._agent

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        return self.load().invoke(*args, **kwargs)

    def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        return self.load().ainvoke(*args, **kwargs)

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        return self.load().stream(*args, **kwargs)

    def astream(self, *args: Any, **kwargs: Any) -> Any:
        return self.load().astream(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.load(), name)


agent = LazyDeepAgent()
