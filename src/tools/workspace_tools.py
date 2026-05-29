from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from src.configs import CONFIG

WORKSPACE_DIR = CONFIG.workspace.root_dir
LATEST_DIR = CONFIG.workspace.latest_dir
RUNS_DIR = CONFIG.workspace.runs_dir

ARTIFACT_FILES = {
    "01_intake_brief.md",
    "02_raw_job_results.md",
    "03_verified_job_results.md",
    "04_resume_match_report.md",
    "05_company_research.md",
    "06_final_job_search_report.md",
}
RUN_ID_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}_\d{6}(?:_\d+)?\Z")

MAX_STATE_REQUIREMENTS = 5
MAX_STATE_REQUIREMENT_CHARS = 160
MAX_STATE_SOURCE_URLS = 8
PLACEHOLDER_TEXT_VALUES = {
    "",
    "n/a",
    "na",
    "none",
    "not available",
    "not specified",
    "unknown",
    "unavailable",
    "unspecified",
}
VERIFICATION_STATUSES_THAT_PROVE_ACTIVE = {"verified"}
VERIFICATION_STATUSES_THAT_PROVE_CLOSED = {"closed"}
VERIFICATION_STATUSES_WITH_UNKNOWN_LIFECYCLE = {
    "access_blocked",
    "login_required",
    "not_verified_backlog",
    "unverified",
}


@tool
def start_workspace_run() -> dict[str, Any]:
    """Create workspace/latest and a run directory."""
    run_id = _new_run_id()
    run_dir = RUNS_DIR / run_id
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=False)
    cleared_artifacts = _clear_latest_artifacts()
    _ensure_state_file()
    return {
        "success": True,
        "run_id": run_id,
        "latest_dir": str(LATEST_DIR),
        "run_dir": str(run_dir),
        "cleared_latest_artifacts": cleared_artifacts,
    }


@tool
def save_workspace_file(file_path: str, content: str) -> dict[str, Any]:
    """Save content to a workspace file, overwriting existing content."""
    path = _resolve_workspace_path(file_path)
    if path is None:
        return {
            "success": False,
            "file_path": file_path,
            "error": "File path must stay inside the workspace directory.",
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return {
        "success": True,
        "file_path": str(path),
        "bytes": len(content.encode("utf-8")),
    }


@tool
def save_job_artifact(run_id: str, file_name: str, content: str) -> dict[str, Any]:
    """Save a job-search artifact to latest/ and the matching runs/<run_id>/ snapshot."""
    if file_name not in ARTIFACT_FILES:
        return {
            "success": False,
            "file_name": file_name,
            "error": f"Unsupported artifact file. Expected one of: {sorted(ARTIFACT_FILES)}",
        }

    run_dir, error = _existing_run_dir_or_error(run_id)
    if error is not None:
        return error
    assert run_dir is not None

    latest_path = LATEST_DIR / file_name
    run_path = run_dir / file_name

    latest_path.parent.mkdir(parents=True, exist_ok=True)
    run_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(content, encoding="utf-8")
    run_path.write_text(content, encoding="utf-8")
    return {
        "success": True,
        "file_name": file_name,
        "latest_path": str(latest_path),
        "run_path": str(run_path),
        "bytes": len(content.encode("utf-8")),
    }


@tool
def update_job_search_state(run_id: str, state_patch_json: str) -> dict[str, Any]:
    """Merge a JSON state patch into latest/job_search_state.json."""
    _run_dir, error = _existing_run_dir_or_error(run_id)
    if error is not None:
        return error

    try:
        patch = _loads_jsonish(state_patch_json)
    except ValueError as error:
        return {"success": False, "error": str(error)}

    if not isinstance(patch, dict):
        return {"success": False, "error": "State patch must be a JSON object."}
    patch = _compact_state_patch(patch)

    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    state = _read_state()
    now = _now_iso()
    state["updated_at"] = now

    if isinstance(patch.get("candidate"), dict):
        state["candidate"] = _merge_dict(state.get("candidate", {}), patch["candidate"])

    if isinstance(patch.get("jobs"), list):
        state["jobs"] = _merge_jobs(state.get("jobs", []), patch["jobs"], now)

    if isinstance(patch.get("companies"), list):
        state["companies"] = _merge_companies(state.get("companies", []), patch["companies"], now)

    _upsert_run(state, run_id, patch.get("run"))
    _write_state(state)

    return {
        "success": True,
        "file_path": str(LATEST_DIR / "job_search_state.json"),
        "job_count": len(state["jobs"]),
        "company_count": len(state["companies"]),
        "run_count": len(state["runs"]),
    }


def _new_run_id() -> str:
    base = datetime.now().astimezone().strftime("%Y-%m-%d_%H%M%S")
    run_id = base
    suffix = 1
    while (RUNS_DIR / run_id).exists():
        suffix += 1
        run_id = f"{base}_{suffix}"
    return run_id


def _clear_latest_artifacts() -> list[str]:
    cleared: list[str] = []
    for file_name in sorted(ARTIFACT_FILES):
        path = LATEST_DIR / file_name
        if not path.is_file() and not path.is_symlink():
            continue
        path.unlink()
        cleared.append(file_name)
    return cleared


def _resolve_run_dir(run_id: str) -> Path | None:
    clean_run_id = str(run_id or "").strip()
    if not RUN_ID_PATTERN.fullmatch(clean_run_id):
        return None

    runs_root = RUNS_DIR.resolve()
    run_dir = (runs_root / clean_run_id).resolve()
    if not _is_relative_to(run_dir, runs_root):
        return None
    return run_dir


def _existing_run_dir_or_error(run_id: str) -> tuple[Path | None, dict[str, Any] | None]:
    run_dir = _resolve_run_dir(run_id)
    if run_dir is None:
        return None, {
            "success": False,
            "run_id": run_id,
            "error": "Invalid run_id. Use the run_id returned by start_workspace_run.",
        }
    if not run_dir.exists() or not run_dir.is_dir():
        return None, {
            "success": False,
            "run_id": run_id,
            "error": "Run directory does not exist. Call start_workspace_run first.",
        }
    return run_dir, None


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _today() -> str:
    return datetime.now().astimezone().date().isoformat()


def _resolve_workspace_path(file_path: str) -> Path | None:
    clean_path = str(file_path or "").strip()
    if not clean_path:
        return None

    container_workspace = CONFIG.docker.container_workspace_dir.rstrip("/")
    if clean_path == container_workspace or clean_path == "workspace":
        path = WORKSPACE_DIR
    elif clean_path.startswith(f"{container_workspace}/"):
        path = WORKSPACE_DIR / clean_path[len(container_workspace) :].lstrip("/")
    elif clean_path.startswith("workspace/"):
        path = WORKSPACE_DIR / clean_path[len("workspace/") :].lstrip("/")
    else:
        raw_path = Path(clean_path).expanduser()
        if raw_path.is_absolute() and _is_relative_to(raw_path.resolve(), WORKSPACE_DIR):
            path = raw_path
        else:
            path = WORKSPACE_DIR / clean_path.lstrip("/")

    path = path.resolve()
    if not _is_relative_to(path, WORKSPACE_DIR):
        return None
    return path


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _ensure_state_file() -> None:
    path = LATEST_DIR / "job_search_state.json"
    if not path.exists():
        _write_state(_empty_state())


def _empty_state() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "updated_at": _now_iso(),
        "candidate": {},
        "jobs": [],
        "companies": [],
        "runs": [],
    }


def _read_state() -> dict[str, Any]:
    path = LATEST_DIR / "job_search_state.json"
    if not path.exists():
        return _empty_state()
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return _empty_state()
    if not isinstance(state, dict):
        return _empty_state()
    state.setdefault("schema_version", 1)
    state.setdefault("updated_at", _now_iso())
    state.setdefault("candidate", {})
    state.setdefault("jobs", [])
    state.setdefault("companies", [])
    state.setdefault("runs", [])
    return state


def _write_state(state: dict[str, Any]) -> None:
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    (LATEST_DIR / "job_search_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _loads_jsonish(value: str) -> Any:
    text = value.strip()
    if not text:
        raise ValueError("JSON input is empty.")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    json_blocks = list(
        re.finditer(r"```json\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    )
    if json_blocks:
        last_error: json.JSONDecodeError | None = None
        for match in json_blocks:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError as error:
                last_error = error
        raise ValueError(f"Invalid JSON fenced block: {last_error}") from last_error

    for match in re.finditer(r"```(?:[a-zA-Z0-9_-]+)?\s*(.*?)```", text, flags=re.DOTALL):
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            continue

    raise ValueError("Input must be valid JSON or contain a JSON fenced block.")


def _merge_dict(base: Any, patch: dict[str, Any]) -> dict[str, Any]:
    result = dict(base) if isinstance(base, dict) else {}
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def _compact_state_patch(patch: dict[str, Any]) -> dict[str, Any]:
    compacted = dict(patch)
    if isinstance(compacted.get("jobs"), list):
        compacted["jobs"] = [
            _compact_job_for_state(job)
            for job in compacted["jobs"]
            if isinstance(job, dict)
        ]
    if isinstance(compacted.get("companies"), list):
        compacted["companies"] = [
            _compact_company_for_state(company)
            for company in compacted["companies"]
            if isinstance(company, dict)
        ]
    return compacted


def _compact_job_for_state(job: dict[str, Any]) -> dict[str, Any]:
    next_job = dict(job)
    next_job.pop("description", None)
    next_job.pop("description_summary", None)
    next_job.pop("html", None)
    next_job.pop("html_preview", None)
    next_job.pop("text_preview", None)
    next_job.pop("raw_text", None)
    next_job.pop("page_text", None)
    next_job["requirements"] = _compact_string_list(
        next_job.get("requirements"),
        limit=MAX_STATE_REQUIREMENTS,
        max_chars=MAX_STATE_REQUIREMENT_CHARS,
    )
    next_job["source_urls"] = _compact_string_list(
        next_job.get("source_urls"),
        limit=MAX_STATE_SOURCE_URLS,
        max_chars=500,
    )
    return next_job


def _compact_company_for_state(company: dict[str, Any]) -> dict[str, Any]:
    next_company = dict(company)
    next_company["summary"] = _truncate_text(next_company.get("summary"), 800)
    next_company["risks"] = _compact_string_list(next_company.get("risks"), limit=6, max_chars=180)
    next_company["interview_prep"] = _compact_string_list(
        next_company.get("interview_prep"),
        limit=6,
        max_chars=180,
    )
    return next_company


def _compact_string_list(value: Any, *, limit: int, max_chars: int) -> list[str]:
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        items = []
    compacted: list[str] = []
    for item in items:
        text = _truncate_text(item, max_chars)
        if text and not _is_placeholder_text(text) and text not in compacted:
            compacted.append(text)
        if len(compacted) >= limit:
            break
    return compacted


def _is_placeholder_text(value: Any) -> bool:
    text = re.sub(r"\s+", " ", str(value or "")).strip().lower().rstrip(".")
    return text in PLACEHOLDER_TEXT_VALUES


def _truncate_text(value: Any, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _merge_jobs(existing_jobs: list[Any], incoming_jobs: list[Any], now: str) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    by_url: dict[str, str] = {}
    for item in existing_jobs:
        if isinstance(item, dict):
            job = _normalize_job(item, now)
            key = _existing_job_key(job, merged, by_url)
            existing = merged.get(key, {})
            merged[key] = _merge_job(existing, job)
            if job.get("canonical_url"):
                by_url[_canonical_url_key(job["canonical_url"])] = key

    for item in incoming_jobs:
        if not isinstance(item, dict):
            continue
        job = _normalize_job(item, now)
        key = _existing_job_key(job, merged, by_url)
        existing = merged.get(key, {})
        next_job = _merge_job(existing, job)
        next_job["job_id"] = existing.get("job_id") or job["job_id"]
        merged[key] = next_job
        if job.get("canonical_url"):
            by_url[_canonical_url_key(job["canonical_url"])] = key

    return list(merged.values())


def _existing_job_key(
    job: dict[str, Any],
    merged: dict[str, dict[str, Any]],
    by_url: dict[str, str],
) -> str:
    canonical_url = _canonical_url_key(job.get("canonical_url", ""))
    if canonical_url and canonical_url in by_url:
        return by_url[canonical_url]
    job_id = str(job["job_id"])
    if job_id in merged:
        return job_id
    duplicate_key = str(job.get("duplicate_key") or "").strip()
    if duplicate_key:
        for candidate_key, candidate in merged.items():
            if str(candidate.get("duplicate_key") or "").strip() == duplicate_key:
                return candidate_key
    return job_id


def _merge_job(existing: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    first_seen = existing.get("first_seen_at") or job.get("first_seen_at") or _today()
    source_urls = _merge_unique(existing.get("source_urls", []), job.get("source_urls", []))
    if job.get("canonical_url"):
        source_urls = _merge_unique(source_urls, [job["canonical_url"]])

    next_job = _merge_job_fields(existing, job)
    next_job["job_id"] = existing.get("job_id") or job["job_id"]
    next_job["first_seen_at"] = first_seen
    next_job["last_seen_at"] = job.get("last_seen_at") or _today()
    next_job["source_urls"] = source_urls
    explicit_status = _status_from_explicit_verification(job)
    if explicit_status is not None:
        if explicit_status == "unknown" and existing.get("status") == "closed":
            next_job["status"] = "closed"
        else:
            next_job["status"] = explicit_status
    else:
        next_job.setdefault("status", "active")
    return next_job


def _merge_job_fields(existing: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    next_job = dict(existing)
    for key, value in job.items():
        if _should_keep_existing_job_value(key, existing.get(key), value):
            continue
        next_job[key] = value
    return next_job


def _should_keep_existing_job_value(key: str, existing_value: Any, incoming_value: Any) -> bool:
    if not _has_meaningful_value(existing_value):
        return False
    if key == "status" and incoming_value == "active" and existing_value != "active":
        return True
    return not _has_meaningful_value(incoming_value)


def _merge_companies(existing_companies: list[Any], incoming_companies: list[Any], now: str) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for item in existing_companies:
        if isinstance(item, dict):
            company = _normalize_company(item, now)
            merged[company["company_id"]] = company

    for item in incoming_companies:
        if not isinstance(item, dict):
            continue
        company = _normalize_company(item, now)
        existing = merged.get(company["company_id"], {})
        next_company = _merge_company_fields(existing, company)
        next_company.setdefault("research_status", "not_started")
        next_company.setdefault("last_researched_at", existing.get("last_researched_at"))
        merged[next_company["company_id"]] = next_company

    return list(merged.values())


def _merge_company_fields(existing: dict[str, Any], company: dict[str, Any]) -> dict[str, Any]:
    next_company = dict(existing)
    for key, value in company.items():
        if _should_keep_existing_company_value(key, existing.get(key), value):
            continue
        next_company[key] = value
    return next_company


def _should_keep_existing_company_value(key: str, existing_value: Any, incoming_value: Any) -> bool:
    if not _has_meaningful_value(existing_value):
        return False
    if key == "research_status" and incoming_value == "not_started" and existing_value != "not_started":
        return True
    return not _has_meaningful_value(incoming_value)


def _normalize_job(job: dict[str, Any], now: str) -> dict[str, Any]:
    next_job = dict(job)
    next_job.setdefault("title", "")
    next_job.setdefault("company", "")
    next_job.setdefault("location", "")
    next_job.setdefault("canonical_url", "")
    next_job.setdefault("source_urls", [])
    next_job["normalized_company"] = next_job.get("normalized_company") or _slug(next_job["company"])
    next_job["normalized_title"] = next_job.get("normalized_title") or _slug(next_job["title"])
    next_job["normalized_location"] = next_job.get("normalized_location") or _slug(next_job["location"])
    dedupe_key = (
        next_job.get("dedupe_key")
        or next_job.get("duplicate_key")
        or f"{next_job['normalized_company']}|{next_job['normalized_title']}|{next_job['normalized_location']}"
    )
    next_job["dedupe_key"] = dedupe_key
    next_job["duplicate_key"] = dedupe_key
    next_job["job_id"] = next_job.get("job_id") or _job_id(next_job)
    next_job.setdefault("first_seen_at", _today())
    next_job.setdefault("last_seen_at", _today())
    next_job.setdefault("status", _status_from_verification(next_job))
    next_job.setdefault("requirements", [])
    next_job.setdefault("confidence", "Unspecified")
    next_job.setdefault("updated_at", now)
    return next_job


def _status_from_verification(job: dict[str, Any]) -> str:
    verification_status = str(job.get("verification_status") or "").strip().lower()
    if verification_status in VERIFICATION_STATUSES_THAT_PROVE_CLOSED:
        return "closed"
    if verification_status in VERIFICATION_STATUSES_THAT_PROVE_ACTIVE:
        return "active"
    if verification_status in VERIFICATION_STATUSES_WITH_UNKNOWN_LIFECYCLE:
        return "unknown"
    return "active"


def _status_from_explicit_verification(job: dict[str, Any]) -> str | None:
    verification_status = str(job.get("verification_status") or "").strip().lower()
    if not verification_status:
        return None
    if verification_status in VERIFICATION_STATUSES_THAT_PROVE_CLOSED:
        return "closed"
    if verification_status in VERIFICATION_STATUSES_THAT_PROVE_ACTIVE:
        return "active"
    if verification_status in VERIFICATION_STATUSES_WITH_UNKNOWN_LIFECYCLE:
        return "unknown"
    return None


def _normalize_company(company: dict[str, Any], now: str) -> dict[str, Any]:
    next_company = dict(company)
    name = str(next_company.get("name") or next_company.get("company") or "").strip()
    next_company["name"] = name
    next_company["company_id"] = next_company.get("company_id") or _slug(name)
    next_company.setdefault("updated_at", now)
    return next_company


def _job_id(job: dict[str, Any]) -> str:
    basis = "|".join(
        [
            str(job.get("normalized_company") or _slug(job.get("company", ""))),
            str(job.get("normalized_title") or _slug(job.get("title", ""))),
            str(job.get("normalized_location") or _slug(job.get("location", ""))),
            str(job.get("canonical_url") or ""),
        ]
    )
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


def _canonical_url_key(value: Any) -> str:
    return str(value or "").strip().rstrip("/")


def _slug(value: Any) -> str:
    text = str(value or "").lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def _merge_unique(first: Any, second: Any) -> list[str]:
    values: list[str] = []
    for source in (first, second):
        if isinstance(source, str):
            items = [source]
        elif isinstance(source, list):
            items = source
        else:
            items = []
        for item in items:
            text = str(item).strip()
            if text and text not in values:
                values.append(text)
    return values


def _has_meaningful_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return not _is_placeholder_text(value)
    if isinstance(value, dict):
        return any(_has_meaningful_value(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_has_meaningful_value(item) for item in value)
    return True


def _upsert_run(state: dict[str, Any], run_id: str, run_patch: Any) -> None:
    run = {
        "run_id": run_id,
        "updated_at": _now_iso(),
        "outputs": {
            "intake_brief": f"workspace/runs/{run_id}/01_intake_brief.md",
            "raw_job_results": f"workspace/runs/{run_id}/02_raw_job_results.md",
            "verified_job_results": f"workspace/runs/{run_id}/03_verified_job_results.md",
            "resume_match_report": f"workspace/runs/{run_id}/04_resume_match_report.md",
            "company_research": f"workspace/runs/{run_id}/05_company_research.md",
            "final_report": f"workspace/runs/{run_id}/06_final_job_search_report.md",
        },
    }
    if isinstance(run_patch, dict):
        run = _merge_dict(run, run_patch)

    runs = [item for item in state.get("runs", []) if isinstance(item, dict) and item.get("run_id") != run_id]
    runs.append(run)
    state["runs"] = runs


WORKSPACE_TOOLS = [
    start_workspace_run,
    save_workspace_file,
    save_job_artifact,
    update_job_search_state,
]
