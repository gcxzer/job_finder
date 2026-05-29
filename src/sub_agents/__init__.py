from langchain_core.language_models.chat_models import BaseChatModel

from src.sub_agents.company_researcher import build_company_researcher
from src.sub_agents.intake_planner import build_intake_planner
from src.sub_agents.job_searcher import build_job_searcher
from src.sub_agents.job_verifier import build_job_verifier
from src.sub_agents.report_writer import build_report_writer
from src.sub_agents.resume_matcher import build_resume_matcher


def build_subagents(model: str | BaseChatModel) -> list[dict]:
    return [
        build_intake_planner(model),
        build_job_searcher(model),
        build_job_verifier(model),
        build_resume_matcher(model),
        build_company_researcher(model),
        build_report_writer(model),
    ]
