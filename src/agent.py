from __future__ import annotations

from deepagents import create_deep_agent

from codex_oauth import CodexOAuthChatModel


def build_deep_agent():
    model = CodexOAuthChatModel()
    return create_deep_agent(
        model=model,
        system_prompt="You are a job search research assistant.",
    )

agent = build_deep_agent()
