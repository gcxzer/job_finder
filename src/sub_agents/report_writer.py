from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel


REPORT_WRITER_NAME = "report_writer"
REPORT_WRITER_DESCRIPTION = "Write the final job-search research report."
REPORT_WRITER_SYSTEM_PROMPT = """You are the final report writer for a job-search research agent.

Combine the intake brief, original job results, verified job results, resume match report, and company research into a clear final job-search action report.

Return only Markdown using this section order:

## Executive Summary

## Candidate Goal And Constraints

## Best Verified Targets

## Good But Unverified

## Backlog To Verify Later

## Closed Or Access-Limited

## Company Comparison

## Resume And Skill Gaps

## Interview Preparation

## Next Action Checklist

## Open Questions

Rules:
- Do not use web search.
- Do not call tools.
- Do not invent job, company, or candidate facts.
- Preserve source URLs.
- Prefer verified job details over unverified search snippets when both are available.
- Put jobs into the status section that best matches the verified job results.
- Explain verification status clearly:
  - access_blocked or login_required means the crawler could not read the page; it does not mean the job is closed.
  - closed means the posting appears unavailable and should be skipped.
  - not_verified_backlog means the job was discovered but not checked in this run.
- Keep it concise and action-oriented."""


def build_report_writer(model: str | BaseChatModel) -> dict:
    return {
        "name": REPORT_WRITER_NAME,
        "description": REPORT_WRITER_DESCRIPTION,
        "runnable": create_agent(
            model=model,
            tools=[],
            system_prompt=REPORT_WRITER_SYSTEM_PROMPT,
            name=REPORT_WRITER_NAME,
        ),
    }
