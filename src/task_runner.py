from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import re
import tomllib
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from src.configs import CONFIG, PROJECT_ROOT
from src.logging_utils import setup_logger
from src.sub_agents.intake_planner import detect_phone_numbers

TASK_CONFIG_ENV_VAR = "JOB_FINDER_TASK_CONFIG"

MAX_LOG_TEXT_CHARS = 1200
MAX_LOG_ITEMS = 10
MAX_LOG_DEPTH = 3
DEFAULT_LOCK_PATH = CONFIG.workspace.root_dir / "job_finder.lock"
DEFAULT_TASK_CONFIG_PATH = PROJECT_ROOT / "src" / "configs" / "job_search.toml"
EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)

class LockAlreadyHeld(RuntimeError):
    """Raised when another scheduled job_finder run is already active."""


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)

    try:
        task = load_job_search_task(args.task_config)
    except (FileNotFoundError, ValueError, tomllib.TOMLDecodeError) as exc:
        raise SystemExit(f"Failed to load job search task config: {exc}") from exc

    if args.no_lock:
        asyncio.run(_run_task(task))
        return

    try:
        with run_lock(args.lock_file):
            asyncio.run(_run_task(task))
    except LockAlreadyHeld as exc:
        logger.warning("Skipped scheduled run: %s", exc)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the job_finder search pipeline once.")
    parser.add_argument(
        "--task-config",
        default=None,
        help=(
            f"Path to a TOML job search config. Defaults to {DEFAULT_TASK_CONFIG_PATH}. "
            f"Can also be set with {TASK_CONFIG_ENV_VAR}."
        ),
    )
    parser.add_argument(
        "--lock-file",
        type=Path,
        default=DEFAULT_LOCK_PATH,
        help=f"Lock file used to skip overlapping scheduled runs. Defaults to {DEFAULT_LOCK_PATH}.",
    )
    parser.add_argument(
        "--no-lock",
        action="store_true",
        help="Run without the overlap-prevention lock.",
    )
    return parser.parse_args(argv)


def load_job_search_task(config_path: str | Path | None = None) -> str:
    path = resolve_task_config_path(config_path)
    values = _load_job_search_config(path)

    if path != DEFAULT_TASK_CONFIG_PATH:
        values = _load_job_search_config(DEFAULT_TASK_CONFIG_PATH) | values

    return build_task(values)


def resolve_task_config_path(config_path: str | Path | None = None) -> Path:
    raw_path = config_path or os.environ.get(TASK_CONFIG_ENV_VAR) or DEFAULT_TASK_CONFIG_PATH
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def build_task(values: Mapping[str, Any], *, search_date: str | None = None) -> str:
    lines = []
    if not _has_date_context(values):
        lines.append(f"search_date: {search_date or _current_date_iso()}")
    for field, value in values.items():
        lines.append(f"{field}: {_format_task_value(value)}")
    return "\n".join(lines)


def _has_date_context(values: Mapping[str, Any]) -> bool:
    return any(key in values for key in ("search_date", "current_date", "run_date"))


def _current_date_iso() -> str:
    return datetime.now().astimezone().date().isoformat()


def _load_job_search_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Job search task config does not exist: {path}")

    with path.open("rb") as config_file:
        data = tomllib.load(config_file)

    values = data.get("job_search", data)
    if not isinstance(values, Mapping):
        raise ValueError(f"Job search task config must contain a table: {path}")
    return dict(values)


def _format_task_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Mapping):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray | str):
        items = []
        for item in value:
            text = _format_task_value(item)
            if text:
                items.append(text)
        return ", ".join(items)
    return str(value)


@contextmanager
def run_lock(lock_path: str | Path = DEFAULT_LOCK_PATH) -> Iterator[Path]:
    path = Path(lock_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise LockAlreadyHeld(f"Another job_finder run is already active: {path}") from exc

        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(f"pid={os.getpid()}\n")
        lock_file.write(f"started_at={datetime.now().astimezone().isoformat()}\n")
        lock_file.flush()

        try:
            yield path
        finally:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


async def _run_task(task: str) -> None:
    from src.deep_agent import agent

    await run_task(task, agent)


async def run_task(task: str, agent: Any) -> None:
    logger.info("USER: %s", _safe_log_text(task))
    try:
        logger.info("Starting agent stream")
        async for event in agent.astream({"messages": [("user", task)]}, config=agent_config):
            logger.info("Received event keys: %s", list(event.keys()))
            for node_name, node_data in event.items():
                if node_data is None or "messages" not in node_data:
                    continue

                msgs = node_data["messages"]
                if not isinstance(msgs, list):
                    msgs = [msgs]

                for msg in msgs:
                    if not isinstance(msg, BaseMessage):
                        continue

                    tool_calls = getattr(msg, "tool_calls", None)
                    if tool_calls:
                        for tool_call in tool_calls:
                            log_tool_call(node_name, tool_call)
                    elif isinstance(msg, ToolMessage):
                        log_tool_output(node_name, msg)
                    elif isinstance(msg, AIMessage) and msg.content and not tool_calls:
                        logger.info("AI[%s]: %s", node_name, _safe_log_text(msg.content))

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        raise
    except Exception:
        logger.exception("Agent run failed")
        raise

def log_tool_call(node_name: str, tool_call: dict) -> None:
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    if tool_name == "task":
        assignee = tool_args.get("assignee") or tool_args.get("subagent_type", "unknown")
        content = tool_args.get("content") or tool_args.get("description", "")
        logger.info("SUB_AGENT_CALL[%s -> %s]: %s", node_name, assignee, _safe_log_text(content))
        return

    logger.info("TOOL_CALL[%s]: %s args=%s", node_name, tool_name, _safe_log_value(tool_args))


def log_tool_output(node_name: str, message: ToolMessage) -> None:
    if message.name == "task":
        logger.info("SUB_AGENT_RESULT[%s]: %s", node_name, _safe_log_text(message.content))
        return

    logger.info("TOOL_OUTPUT[%s:%s]: %s", node_name, message.name, _safe_log_text(message.content))


def _safe_log_text(value: Any, *, max_chars: int = MAX_LOG_TEXT_CHARS) -> str:
    text = _redact_pii(_content_to_text(value))
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars].rstrip()}... [truncated {omitted} chars]"


def _safe_log_value(value: Any, *, depth: int = 0) -> Any:
    if isinstance(value, str):
        return _safe_log_text(value)
    if isinstance(value, bytes | bytearray):
        return f"<{len(value)} bytes>"
    if depth >= MAX_LOG_DEPTH:
        return f"<{type(value).__name__} omitted>"
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        items = list(value.items())
        for key, item_value in items[:MAX_LOG_ITEMS]:
            result[str(key)] = _safe_log_value(item_value, depth=depth + 1)
        if len(items) > MAX_LOG_ITEMS:
            result["..."] = f"{len(items) - MAX_LOG_ITEMS} more keys"
        return result
    if isinstance(value, Sequence) and not isinstance(value, str):
        items = list(value)
        result = [_safe_log_value(item, depth=depth + 1) for item in items[:MAX_LOG_ITEMS]]
        if len(items) > MAX_LOG_ITEMS:
            result.append(f"... {len(items) - MAX_LOG_ITEMS} more items")
        return result
    return value


def _content_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        chunks: list[str] = []
        for item in value:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, Mapping):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    chunks.append(text)
            else:
                chunks.append(str(item))
        return "\n".join(chunks)
    return str(value)


def _redact_pii(text: str) -> str:
    text = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", text)
    return _redact_phone_numbers(text)


def _redact_phone_numbers(text: str) -> str:
    matches = detect_phone_numbers(text)
    if not matches:
        return text

    chunks: list[str] = []
    cursor = 0
    for match in sorted(matches, key=lambda item: int(item["start"])):
        start = int(match["start"])
        end = int(match["end"])
        if start < cursor:
            continue
        chunks.append(text[cursor:start])
        chunks.append("[REDACTED_PHONE]")
        cursor = end
    chunks.append(text[cursor:])
    return "".join(chunks)


agent_config = {
    "recursion_limit": 80,
    "configurable": {"thread_id": "job_finder_orchestrator"},
}
logger = setup_logger()

