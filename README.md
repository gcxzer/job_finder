# job_finder

DeepAgents-based job-search research assistant. The agent takes a job-search
profile, finds current postings, verifies job pages, compares roles against the
candidate profile or resume, researches top companies, and writes a final action
report.

The project is designed for repeatable local runs: outputs are saved under
`workspace/`, execution tools run in Docker, and the CLI is safe to schedule with
cron.

## What It Does

The main agent coordinates specialist subagents in this order:

1. Intake planning: parse the search profile and optional PDF resume.
2. Job search: use web search to discover and rank candidate postings.
3. Job verification: fetch job pages, extract evidence, and mark verification status.
4. Resume matching: score verified jobs against known candidate evidence.
5. Company research: research the top matched companies.
6. Final report: write a concise, actionable job-search report.

Each run writes both "latest" artifacts and a timestamped snapshot so scheduled
runs do not overwrite historical results.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Docker Desktop or a running Docker daemon
- Chromium browser binaries for Playwright

## Setup From Clone

Clone the project and enter the workspace:

```bash
git clone https://github.com/gcxzer/job_finder.git
cd job_finder
```

Install dependencies and Playwright's Chromium browser:

```bash
uv sync
uv run playwright install chromium
```

Start Docker Desktop before running the agent. DeepAgents shell execution runs
inside a Docker container, and the local `workspace/` directory is mounted at
`/workspace`.


## Configure Search

The default scheduled search profile lives in `src/configs/job_search.toml`:

```toml
[job_search]
resume_pdf_path = ""
target_roles = []
target_locations = []
remote_preference = ""
industries = []
company_preferences = []
salary_expectation = ""
start_date_or_timeline = ""
work_authorization_or_visa = ""
must_have_constraints = []
nice_to_have = []
excluded_roles_or_companies = []
notes = ""
```

Use `resume_pdf_path` only when you want resume-based analysis. Put PDFs under
`workspace/` and reference them with a workspace path such as
`workspace/resumes/resume.pdf` or `/workspace/resumes/resume.pdf`; the document
tool rejects paths outside the workspace. Partial custom profiles are supported:
values from a custom TOML file are merged over the default profile.

Run with a different profile by creating another TOML file, then passing
`--task-config` or setting `JOB_FINDER_TASK_CONFIG`:

```bash
cp src/configs/job_search.toml src/configs/my_job_search.toml
uv run job-finder-run --task-config src/configs/my_job_search.toml
```

## Run Locally

Run the full pipeline once:

```bash
uv run job-finder-run
```

Run without the overlap-prevention lock:

```bash
uv run job-finder-run --no-lock
```

The legacy Python entrypoint still works:

```bash
uv run python main.py
```

For LangGraph development, `langgraph.json` exposes graph id `agent`:

```bash
uv run langgraph dev
```

To connect the local Agent Chat UI, keep the LangGraph server running on port
`2024`:

```bash
uv run langgraph dev --port 2024
```

In another terminal, start the UI:

```bash
git clone https://github.com/langchain-ai/agent-chat-ui.git
cd agent-chat-ui
pnpm install
pnpm dev
```

Then open it with `assistantId=agent`:

```text
http://localhost:3000/?apiUrl=http://localhost:2024&assistantId=agent
```

## Outputs

The agent writes the latest run to `workspace/latest/`:

```text
workspace/latest/
  01_intake_brief.md
  02_raw_job_results.md
  03_verified_job_results.md
  04_resume_match_report.md
  05_company_research.md
  06_final_job_search_report.md
  job_search_state.json
```

Each run also gets a snapshot under:

```text
workspace/runs/<run_id>/
```

Runtime logs are written to:

```text
runs/logs/job_finder_<timestamp>.log
```

During verification, fetched pages and generated crawler helpers may also be
stored under `workspace/page_cache/` and `workspace/crawlers/`.

## Configuration Reference

Settings can be loaded from `.env` or environment variables. All project-specific
variables use the `JOB_FINDER_` prefix.

| Variable | Default | Purpose |
| --- | --- | --- |
| `JOB_FINDER_MODEL_PROVIDER` | `codex_oauth` | Model backend/provider. The default uses local Codex OAuth credentials; other LangChain-compatible providers can be configured as needed. |
| `JOB_FINDER_MODEL_NAME` | `gpt-5.5` | Model name passed to the selected provider. |
| `JOB_FINDER_REASONING_EFFORT` | `xhigh` | Reasoning effort passed to providers that support it. |
| `JOB_FINDER_CODEX_AUTH_PATH` | `.codex_oauth/auth/codex.json` | Credential file used only by the default `codex_oauth` provider. |
| `JOB_FINDER_TASK_CONFIG` | `src/configs/job_search.toml` | TOML profile for scheduled runs. |
| `JOB_FINDER_WORKSPACE_DIR` | `workspace` | Root directory for artifacts, snapshots, caches, and locks. |
| `JOB_FINDER_LOG_DIR` | `runs/logs` | Runtime log directory. |
| `JOB_FINDER_DOCKER_IMAGE` | `python:3.12-slim` | Docker image used for agent shell execution. |
| `JOB_FINDER_CONTAINER_WORKSPACE_DIR` | `/workspace` | Mount path inside the Docker container. |
| `JOB_FINDER_DOCKER_MEMORY_LIMIT` | `1g` | Docker memory limit. |
| `JOB_FINDER_DOCKER_CPU_QUOTA` | `100000` | Docker CPU quota. |
| `JOB_FINDER_DOCKER_NETWORK_DISABLED` | `false` | Disable network access inside the execution container. Generated crawler URLs are still checked before execution. |
| `JOB_FINDER_MAX_DISCOVERED_JOBS` | `40` | Maximum jobs to discover before dedupe. |
| `JOB_FINDER_MAX_DETAILED_JOBS` | `20` | Maximum detailed ranked jobs in raw search output. |
| `JOB_FINDER_MAX_VERIFIED_JOBS` | `20` | Maximum jobs to fetch and verify per run. |
| `JOB_FINDER_COMPANY_RESEARCH_TOP_N` | `3` | Number of top targets to research. |

## Scheduled Runs With cron

`job-finder-run` is cron-friendly. It loads the TOML task config, runs the agent
once, and uses `workspace/job_finder.lock` to skip overlapping runs when a
previous search is still active.

Example crontab entry for weekdays at 09:00:

```cron
0 9 * * 1-5 cd /path/to/job_finder && mkdir -p runs/logs && /opt/homebrew/bin/uv run job-finder-run >> runs/logs/cron.log 2>&1
```

If `uv` is installed somewhere else, replace `/opt/homebrew/bin/uv` with the
output of:

```bash
command -v uv
```

To use a custom profile:

```cron
0 9 * * 1-5 cd /path/to/job_finder && mkdir -p runs/logs && /opt/homebrew/bin/uv run job-finder-run --task-config src/configs/job_search.toml >> runs/logs/cron.log 2>&1
```

## Development

Run the test suite:

```bash
uv run python -m unittest discover
```

Useful implementation files:

```text
job_finder/
  langgraph.json              # LangGraph server config; exposes graph id "agent"
  pyproject.toml              # Python project, dependencies, and CLI entrypoints
  main.py                     # Thin legacy entrypoint to src.task_runner
  src/
    configs/                  # Runtime settings and default search profile
    deep_agent.py             # DeepAgents graph exported as `agent`
    docker_backend.py         # Docker execution backend used by DeepAgents
    logging_utils.py          # File and console logging setup
    task_runner.py            # CLI, task profile loading, cron lock, stream logging
    codex_oauth/              # Optional Codex OAuth LangChain chat model
    sub_agents/               # Intake, search, verification, matching, research, report agents
    tools/                    # Workspace, document, page extraction, and crawler tools
  tests/                      # unittest coverage for core helpers and tools
```

## Troubleshooting

- Docker errors: make sure Docker Desktop is running and that the configured
  image can be pulled.
- Default provider auth errors: when using `JOB_FINDER_MODEL_PROVIDER=codex_oauth`,
  run `uv run codex-oauth-login` again, or set `JOB_FINDER_CODEX_AUTH_PATH`.
- Model auth errors: check the credentials required by the configured
  `JOB_FINDER_MODEL_PROVIDER`.
- Browser extraction errors: rerun `uv run playwright install chromium`.
- Scheduled run skipped: check whether `workspace/job_finder.lock` exists
  because another run is active.
- Sparse match scores: provide a `resume_pdf_path` or more candidate details in
  the TOML profile.
