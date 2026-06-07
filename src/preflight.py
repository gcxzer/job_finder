from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.configs import CONFIG, AppConfig

try:
    import docker
except ImportError:
    docker = None


DEFAULT_DOCKER_DESKTOP_SOCKET = Path.home() / ".docker" / "run" / "docker.sock"
DEFAULT_DOCKER_CONFIG_DIR = Path.home() / ".docker"
COMMON_CRON_PATH_ENTRIES = ("/usr/local/bin", "/opt/homebrew/bin")


class PreflightError(RuntimeError):
    """Raised when the job cannot safely start."""


@dataclass(frozen=True)
class PreflightResult:
    messages: tuple[str, ...]


def run_preflight_checks(config: AppConfig = CONFIG) -> PreflightResult:
    """Verify local dependencies before the expensive agent run starts."""

    messages: list[str] = []
    messages.extend(_prepare_docker_environment())
    messages.append(_check_writable_directory(config.workspace.logs_dir, "log"))
    messages.append(_check_writable_directory(config.workspace.root_dir, "workspace"))
    messages.extend(_check_docker(config))
    return PreflightResult(tuple(messages))


def _prepare_docker_environment() -> list[str]:
    messages: list[str] = []

    added_path_entries = _ensure_path_entries(COMMON_CRON_PATH_ENTRIES)
    if added_path_entries:
        messages.append(f"added Docker helper paths to PATH: {', '.join(added_path_entries)}")

    if not os.environ.get("DOCKER_HOST") and DEFAULT_DOCKER_DESKTOP_SOCKET.exists():
        os.environ["DOCKER_HOST"] = f"unix://{DEFAULT_DOCKER_DESKTOP_SOCKET}"
        messages.append(f"set DOCKER_HOST={os.environ['DOCKER_HOST']}")

    if not os.environ.get("DOCKER_CONFIG") and DEFAULT_DOCKER_CONFIG_DIR.exists():
        os.environ["DOCKER_CONFIG"] = str(DEFAULT_DOCKER_CONFIG_DIR)
        messages.append(f"set DOCKER_CONFIG={os.environ['DOCKER_CONFIG']}")

    return messages


def _ensure_path_entries(paths: tuple[str, ...]) -> list[str]:
    current = [entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry]
    added: list[str] = []

    for path in reversed(paths):
        if Path(path).is_dir() and path not in current:
            current.insert(0, path)
            added.append(path)

    if added:
        os.environ["PATH"] = os.pathsep.join(current)
    return list(reversed(added))


def _check_writable_directory(path: Path, label: str) -> str:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe_path = path / f".job_finder_preflight_{os.getpid()}"
        probe_path.write_text("ok\n", encoding="utf-8")
        probe_path.unlink()
    except OSError as error:
        raise PreflightError(
            f"Cannot write to {label} directory {path}: {error}. "
            "On macOS, grant /usr/sbin/cron Full Disk Access if this runs from cron."
        ) from error

    return f"{label} directory writable: {path}"


def _check_docker(config: AppConfig) -> list[str]:
    if docker is None:
        raise PreflightError("Python package `docker` is not installed. Run `uv sync` or install dependencies.")

    _check_docker_credential_helpers()

    try:
        client = docker.from_env()
    except Exception as error:
        raise PreflightError(_docker_unavailable_message(error)) from error

    try:
        version = client.version().get("Version", "unknown")
    except Exception as error:
        raise PreflightError(_docker_unavailable_message(error)) from error

    messages = [f"Docker daemon reachable: {version}"]
    image = config.docker.image

    if config.docker.container_id:
        _check_existing_container(client, config.docker.container_id)
        messages.append(f"Docker container available: {config.docker.container_id}")
        return messages

    try:
        client.images.get(image)
        messages.append(f"Docker image available: {image}")
    except Exception as error:
        if not _is_image_not_found(error):
            raise PreflightError(f"Failed to inspect Docker image {image}: {error}") from error
        try:
            client.images.pull(image)
        except Exception as pull_error:
            raise PreflightError(
                f"Docker image {image} is missing and could not be pulled: {pull_error}. "
                f"Try: DOCKER_HOST={os.environ.get('DOCKER_HOST', '<docker-host>')} docker pull {image}"
            ) from pull_error
        messages.append(f"Docker image pulled: {image}")

    return messages


def _check_existing_container(client: Any, container_id: str) -> None:
    try:
        client.containers.get(container_id)
    except Exception as error:
        raise PreflightError(f"Configured Docker container is not available: {container_id}: {error}") from error


def _check_docker_credential_helpers() -> None:
    config_dir = Path(os.environ.get("DOCKER_CONFIG", DEFAULT_DOCKER_CONFIG_DIR)).expanduser()
    helpers = _configured_credential_helpers(config_dir / "config.json")
    missing = [f"docker-credential-{helper}" for helper in helpers if shutil.which(f"docker-credential-{helper}") is None]
    if not missing:
        return

    raise PreflightError(
        "Docker credential helper is missing from PATH: "
        f"{', '.join(missing)}. "
        "Cron usually needs PATH to include /usr/local/bin and /opt/homebrew/bin."
    )


def _configured_credential_helpers(config_path: Path) -> list[str]:
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return []

    helpers: set[str] = set()
    creds_store = data.get("credsStore")
    if isinstance(creds_store, str) and creds_store.strip():
        helpers.add(creds_store.strip())

    cred_helpers = data.get("credHelpers")
    if isinstance(cred_helpers, dict):
        helpers.update(str(helper).strip() for helper in cred_helpers.values() if str(helper).strip())

    return sorted(helpers)


def _docker_unavailable_message(error: Exception) -> str:
    docker_host = os.environ.get("DOCKER_HOST") or "Docker default socket"
    return (
        f"Docker daemon is not reachable via {docker_host}: {error}. "
        "Start Docker Desktop before running job-finder-run."
    )


def _is_image_not_found(error: Exception) -> bool:
    error_type = type(error).__name__
    message = str(error).lower()
    return error_type in {"ImageNotFound", "NotFound"} or "no such image" in message or "not found" in message
