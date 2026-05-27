# job_finder

DeepAgents job-search agent experiments.

## Setup From Clone

Clone the project and enter the workspace:

```bash
git clone https://github.com/gcxzer/job_finder.git
cd job_finder
```

Install the Python dependencies:

```bash
uv sync
```

## Optional: Codex OAuth Model

The current `src/agent.py` uses `CodexOAuthChatModel()` by default. If you keep
that default, connect once before chatting:

```bash
uv run codex-oauth-login
```

Credentials are stored at `.codex_oauth/auth/codex.json` by default. Override
with `CODEX_OAUTH_AUTH_PATH=/path/to/codex.json`.

## Run With Agent Chat UI

In terminal 1, start the LangGraph server from this project:

```bash
uv run langgraph dev --port 2024
```

Keep this terminal running.

In terminal 2, clone and run Agent Chat UI:

```bash
git clone https://github.com/langchain-ai/agent-chat-ui.git
cd agent-chat-ui
pnpm install
pnpm dev
```

Then open:

```text
http://localhost:3000/?apiUrl=http://localhost:2024&assistantId=agent
```

## Current Structure

```text
job_finder/
  langgraph.json              # LangGraph server config; exposes graph id "agent"
  pyproject.toml              # Python project, dependencies, and CLI entrypoints
  src/
    agent.py                  # DeepAgents graph exported as `agent`
    codex_oauth/              # Optional Codex OAuth LangChain chat model
```

Agent Chat UI should use `assistantId=agent`.
