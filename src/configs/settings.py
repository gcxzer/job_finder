from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    model_name: str
    reasoning_effort: str
    codex_auth_path: Path


class DockerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    image: str
    container_id: str | None
    container_workspace_dir: str
    auto_remove: bool
    network_disabled: bool
    memory_limit: str
    cpu_quota: int


class SearchConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_discovered_jobs: int
    max_detailed_jobs: int
    max_verified_jobs: int
    company_research_top_n: int


class WorkspaceConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    root_dir: Path
    logs_dir: Path

    @property
    def latest_dir(self) -> Path:
        return self.root_dir / "latest"

    @property
    def runs_dir(self) -> Path:
        return self.root_dir / "runs"

    @property
    def page_cache_dir(self) -> Path:
        return self.root_dir / "page_cache"

    @property
    def crawlers_dir(self) -> Path:
        return self.root_dir / "crawlers"


class AppConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: ModelConfig
    docker: DockerConfig
    search: SearchConfig
    workspace: WorkspaceConfig


class EnvSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="JOB_FINDER_",
        extra="ignore",
        populate_by_name=True,
    )

    # Model provider and authentication.
    model_provider: str = "codex_oauth"
    model_name: str = "gpt-5.5"
    reasoning_effort: str = "xhigh"
    codex_auth_path: Path = Field(
        default=PROJECT_ROOT / ".codex_oauth" / "auth" / "codex.json",
        validation_alias=AliasChoices("JOB_FINDER_CODEX_AUTH_PATH", "CODEX_OAUTH_AUTH_PATH"),
    )

    # Docker execution backend.
    docker_image: str = "python:3.12-slim"
    docker_container_id: str | None = None
    container_workspace_dir: str = "/workspace"
    docker_auto_remove: bool = True
    docker_network_disabled: bool = False
    docker_memory_limit: str = "1g"
    docker_cpu_quota: int = 100000

    # Search funnel limits.
    max_discovered_jobs: int = Field(default=40, ge=1)
    max_detailed_jobs: int = Field(default=20, ge=1)
    max_verified_jobs: int = Field(default=20, ge=1)
    company_research_top_n: int = Field(default=3, ge=1)

    # Local output paths.
    workspace_dir: Path = PROJECT_ROOT / "workspace"
    log_dir: Path = PROJECT_ROOT / "runs" / "logs"

    @field_validator("codex_auth_path", "workspace_dir", "log_dir", mode="after")
    @classmethod
    def resolve_path(cls, value: Path) -> Path:
        path = value.expanduser()
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    def to_app_config(self) -> AppConfig:
        return AppConfig(
            model=ModelConfig(
                provider=self.model_provider,
                model_name=self.model_name,
                reasoning_effort=self.reasoning_effort,
                codex_auth_path=self.codex_auth_path,
            ),
            docker=DockerConfig(
                image=self.docker_image,
                container_id=self.docker_container_id,
                container_workspace_dir=self.container_workspace_dir,
                auto_remove=self.docker_auto_remove,
                network_disabled=self.docker_network_disabled,
                memory_limit=self.docker_memory_limit,
                cpu_quota=self.docker_cpu_quota,
            ),
            search=SearchConfig(
                max_discovered_jobs=self.max_discovered_jobs,
                max_detailed_jobs=self.max_detailed_jobs,
                max_verified_jobs=self.max_verified_jobs,
                company_research_top_n=self.company_research_top_n,
            ),
            workspace=WorkspaceConfig(
                root_dir=self.workspace_dir,
                logs_dir=self.log_dir,
            ),
        )


def load_config(settings: EnvSettings | None = None) -> AppConfig:
    return (settings or EnvSettings()).to_app_config()


CONFIG = load_config()
