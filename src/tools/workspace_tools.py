from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

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
FINAL_REPORT_FILE_NAME = "06_final_job_search_report.md"
FINAL_REPORT_URL_SOURCE_SECTIONS = (
    ("03_verified_job_results.md", "Verified Job State JSON"),
    ("02_raw_job_results.md", "Job State JSON"),
    ("04_resume_match_report.md", "Match State JSON"),
)
RUN_ID_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}_\d{6}(?:_\d+)?\Z")

MAX_STATE_SOURCE_URLS = 8
MAX_DEDUPE_STATE_JOBS = 60
DEDUPE_STATUS_RANK = {
    "active": 0,
    "unknown": 0,
    "closed": 1,
}
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
REPORT_URL_PLACEHOLDER_VALUES = {
    "url not supplied",
}
REPORT_MATCH_STOPWORDS = {
    "a",
    "ag",
    "an",
    "and",
    "at",
    "co",
    "company",
    "deutschland",
    "d",
    "exact",
    "f",
    "germany",
    "gmbh",
    "in",
    "kg",
    "ltd",
    "m",
    "not",
    "role",
    "supplied",
    "the",
    "title",
    "w",
    "x",
}
VERIFICATION_STATUSES_THAT_PROVE_ACTIVE = {"verified"}
VERIFICATION_STATUSES_THAT_PROVE_CLOSED = {"closed"}
VERIFICATION_STATUSES_WITH_UNKNOWN_LIFECYCLE = {
    "access_blocked",
    "login_required",
    "not_verified_backlog",
    "unverified",
}
TRACKING_QUERY_PARAMS = {
    "fbclid",
    "gclid",
    "gh_src",
    "igshid",
    "li_fat_id",
    "mc_cid",
    "mc_eid",
    "mkt_tok",
    "msclkid",
    "ref",
    "ref_src",
    "referrer",
    "source",
    "src",
    "trk",
    "yclid",
}
TRACKING_QUERY_PREFIXES = ("utm_",)


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
    if file_name == FINAL_REPORT_FILE_NAME:
        content = _repair_final_report_urls(content, run_dir)

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
    """Merge job dedupe keys into latest/job_search_state.json."""
    _run_dir, error = _existing_run_dir_or_error(run_id)
    if error is not None:
        return error

    try:
        patch = _loads_jsonish(state_patch_json)
    except ValueError as error:
        return {"success": False, "error": str(error)}

    if not isinstance(patch, dict):
        return {"success": False, "error": "State patch must be a JSON object."}
    return _merge_job_search_state_patch(patch)


@tool
def update_job_search_state_from_artifact(
    run_id: str,
    file_name: str,
    state_section_heading: str = "",
) -> dict[str, Any]:
    """Extract a state JSON block from a saved artifact and merge it into the job dedupe index."""
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

    artifact_path = run_dir / file_name
    if not artifact_path.exists() or not artifact_path.is_file():
        return {
            "success": False,
            "file_name": file_name,
            "run_id": run_id,
            "error": "Artifact file does not exist. Save the artifact before updating state.",
        }

    content = artifact_path.read_text(encoding="utf-8")
    try:
        patch = _loads_jsonish_from_markdown_section(content, state_section_heading)
    except ValueError as error:
        return {
            "success": False,
            "file_name": file_name,
            "run_id": run_id,
            "state_section_heading": state_section_heading,
            "error": str(error),
        }

    if not isinstance(patch, dict):
        return {"success": False, "error": "State patch must be a JSON object."}
    if not isinstance(patch.get("jobs"), list):
        return {"success": False, "error": "Extracted state patch must include a jobs list."}

    result = _merge_job_search_state_patch(patch)
    result["source_file"] = str(artifact_path)
    result["state_section_heading"] = state_section_heading
    return result


def _merge_job_search_state_patch(patch: dict[str, Any]) -> dict[str, Any]:
    patch = _compact_state_patch(patch)
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    state = _read_state()
    now = _now_iso()

    if isinstance(patch.get("jobs"), list):
        state["jobs"] = _merge_jobs(state.get("jobs", []), patch["jobs"], now)

    _write_state(state)

    return {
        "success": True,
        "file_path": str(LATEST_DIR / "job_search_state.json"),
        "job_count": len(state["jobs"]),
        "merged_job_count": len(patch.get("jobs", [])) if isinstance(patch.get("jobs"), list) else 0,
    }


@tool
def read_job_search_dedupe_state(limit: int = MAX_DEDUPE_STATE_JOBS) -> dict[str, Any]:
    """Read compact job dedupe keys from latest/job_search_state.json."""
    state = _read_state()
    jobs = state.get("jobs", [])
    if not isinstance(jobs, list):
        jobs = []

    clean_limit = max(0, min(int(limit or 0), MAX_DEDUPE_STATE_JOBS))
    selected_jobs = _select_dedupe_jobs(jobs, clean_limit)
    return {
        "success": True,
        "file_path": str(LATEST_DIR / "job_search_state.json"),
        "job_count": len(jobs),
        "returned_job_count": len(selected_jobs),
        "jobs": selected_jobs,
        "dedupe_brief": _dedupe_brief(selected_jobs),
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
        "jobs": [],
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
    jobs = state.get("jobs")
    if not isinstance(jobs, list):
        jobs = []
    now = _now_iso()
    return {"jobs": [_job_dedupe_record(_normalize_job(job, now)) for job in jobs if isinstance(job, dict)]}


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


def _loads_jsonish_from_markdown_section(markdown: str, heading: str = "") -> Any:
    clean_heading = str(heading or "").strip().lstrip("#").strip()
    if not clean_heading:
        return _loads_jsonish(markdown)

    heading_pattern = re.compile(
        rf"^#{{1,6}}\s+{re.escape(clean_heading)}\s*$",
        flags=re.MULTILINE,
    )
    match = heading_pattern.search(markdown)
    if match is None:
        raise ValueError(f'Markdown section heading "{clean_heading}" was not found.')

    section_start = match.end()
    next_heading = re.search(r"^#{1,6}\s+", markdown[section_start:], flags=re.MULTILINE)
    section = (
        markdown[section_start : section_start + next_heading.start()]
        if next_heading is not None
        else markdown[section_start:]
    )
    return _loads_jsonish(section)


def _repair_final_report_urls(content: str, run_dir: Path) -> str:
    records = _load_report_url_records(run_dir)
    if not records or "URL not supplied" not in content:
        return content

    repaired_lines: list[str] = []
    header_cells: list[str] = []
    header_indexes: dict[str, int] = {}

    for line in content.splitlines(keepends=True):
        line_body = line.removesuffix("\n")
        newline = "\n" if line.endswith("\n") else ""
        cells = _split_markdown_table_row(line_body)
        if not cells:
            header_cells = []
            header_indexes = {}
            repaired_lines.append(line)
            continue

        if "url" in {_normalize_table_header(cell) for cell in cells}:
            header_cells = cells
            header_indexes = {
                _normalize_table_header(cell): index
                for index, cell in enumerate(header_cells)
            }
            repaired_lines.append(line)
            continue

        if _is_markdown_table_separator(cells):
            repaired_lines.append(line)
            continue

        url_index = header_indexes.get("url")
        if url_index is None or url_index >= len(cells):
            repaired_lines.append(line)
            continue

        if _normalize_report_url_placeholder(cells[url_index]) not in REPORT_URL_PLACEHOLDER_VALUES:
            repaired_lines.append(line)
            continue

        role = _cell_by_header(cells, header_indexes, "role")
        company = _cell_by_header(cells, header_indexes, "company")
        status = _cell_by_header(cells, header_indexes, "verification status")
        match = _best_report_url_record(records, role=role, company=company, status=status)
        if match is None:
            repaired_lines.append(line)
            continue

        cells[url_index] = _markdown_link("Job page", match["url"])
        repaired_lines.append(_join_markdown_table_row(cells) + newline)

    return "".join(repaired_lines)


def _load_report_url_records(run_dir: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for source_index, (file_name, heading) in enumerate(FINAL_REPORT_URL_SOURCE_SECTIONS):
        path = run_dir / file_name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        try:
            patch = _loads_jsonish_from_markdown_section(text, heading)
        except ValueError:
            patch = None
        if isinstance(patch, dict):
            jobs = patch.get("jobs")
            if isinstance(jobs, list):
                for job_index, job in enumerate(jobs):
                    if not isinstance(job, dict):
                        continue
                    url = _preferred_report_job_url(job)
                    if not url:
                        continue
                    records.append(
                        {
                            "title": str(job.get("title") or ""),
                            "company": str(job.get("company") or ""),
                            "status": str(job.get("verification_status") or ""),
                            "url": url,
                            "source_index": str(source_index),
                            "job_index": str(job_index),
                        }
                    )
        records.extend(_load_markdown_report_url_records(text, source_index=source_index))
    return records


def _load_markdown_report_url_records(text: str, *, source_index: int) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for line_index, line in enumerate(text.splitlines()):
        match = re.search(r"^\s*[-*]\s+Job:\s*(?P<before>.*?)\s*(?P<url>https?://\S+)", line)
        if match is None:
            continue
        url = _clean_report_url(match.group("url").rstrip(".,;"))
        if not url:
            continue
        parts = [
            part.strip()
            for part in re.split(r"\s+—\s+", match.group("before").strip())
            if part.strip()
        ]
        status_match = re.search(r"verification_status:\s*([a-zA-Z_]+)", line)
        records.append(
            {
                "title": parts[0] if parts else "",
                "company": parts[1] if len(parts) > 1 else "",
                "status": status_match.group(1) if status_match else "",
                "url": url,
                "source_index": str(source_index),
                "job_index": str(line_index),
            }
        )
    return records


def _preferred_report_job_url(job: dict[str, Any]) -> str:
    for field_name in ("apply_url", "final_url", "canonical_url", "url"):
        url = _clean_report_url(job.get(field_name))
        if url:
            return url

    source_urls = job.get("source_urls")
    if isinstance(source_urls, str):
        return _clean_report_url(source_urls)
    if isinstance(source_urls, list):
        for item in source_urls:
            url = _clean_report_url(item)
            if url:
                return url
    return ""


def _clean_report_url(value: Any) -> str:
    text = str(value or "").strip()
    if not text or _is_placeholder_text(text):
        return ""
    try:
        parsed = urlsplit(text)
    except ValueError:
        return ""
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return ""
    return text


def _markdown_link(label: str, url: str) -> str:
    return f"[{label}]({url})"


def _split_markdown_table_row(line: str) -> list[str]:
    if not line.lstrip().startswith("|"):
        return []
    text = line.strip()
    if text.startswith("|"):
        text = text[1:]
    if text.endswith("|"):
        text = text[:-1]
    return [cell.strip() for cell in re.split(r"(?<!\\)\|", text)]


def _join_markdown_table_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _is_markdown_table_separator(cells: list[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells)


def _normalize_table_header(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _normalize_report_url_placeholder(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower().rstrip("."))


def _cell_by_header(cells: list[str], header_indexes: dict[str, int], header: str) -> str:
    index = header_indexes.get(header)
    if index is None or index >= len(cells):
        return ""
    return cells[index]


def _best_report_url_record(
    records: list[dict[str, str]],
    *,
    role: str,
    company: str,
    status: str,
) -> dict[str, str] | None:
    best_record: dict[str, str] | None = None
    best_score = 0.0
    for record in records:
        score = _report_url_match_score(record, role=role, company=company, status=status)
        if score > best_score:
            best_record = record
            best_score = score

    return best_record if best_score >= 0.65 else None


def _report_url_match_score(
    record: dict[str, str],
    *,
    role: str,
    company: str,
    status: str,
) -> float:
    role_score = _token_similarity(role, record.get("title", ""))
    company_score = _token_similarity(company, record.get("company", ""))
    company_is_placeholder = _is_placeholder_text(company)
    if company_is_placeholder:
        score = _token_jaccard(role, record.get("title", ""))
    else:
        score = (role_score * 0.65) + (company_score * 0.35)

    row_status = _normalize_report_status(status)
    record_status = _normalize_report_status(record.get("status", ""))
    if row_status and record_status and row_status == record_status:
        score += 0.12

    source_index = int(record.get("source_index") or 0)
    job_index = int(record.get("job_index") or 0)
    return score - (source_index * 0.001) - (job_index * 0.00001)


def _token_similarity(left: str, right: str) -> float:
    left_tokens = _match_tokens(left)
    right_tokens = _match_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = left_tokens & right_tokens
    if not overlap:
        return 0.0
    containment = len(overlap) / min(len(left_tokens), len(right_tokens))
    jaccard = len(overlap) / len(left_tokens | right_tokens)
    return max(containment, jaccard)


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = _match_tokens(left)
    right_tokens = _match_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = left_tokens & right_tokens
    if not overlap:
        return 0.0
    return len(overlap) / len(left_tokens | right_tokens)


def _match_tokens(value: str) -> set[str]:
    slug = _slug(value)
    if not slug:
        return set()
    return {
        token
        for token in slug.split("_")
        if len(token) > 1 and token not in REPORT_MATCH_STOPWORDS
    }


def _normalize_report_status(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip().lower())
    if text.startswith("verified"):
        return "verified"
    if text in VERIFICATION_STATUSES_WITH_UNKNOWN_LIFECYCLE | VERIFICATION_STATUSES_THAT_PROVE_CLOSED:
        return text
    return text


def _merge_dict(base: Any, patch: dict[str, Any]) -> dict[str, Any]:
    result = dict(base) if isinstance(base, dict) else {}
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def _compact_state_patch(patch: dict[str, Any]) -> dict[str, Any]:
    compacted: dict[str, Any] = {}
    if isinstance(patch.get("jobs"), list):
        compacted["jobs"] = [
            _compact_job_for_state(job)
            for job in patch["jobs"]
            if isinstance(job, dict)
        ]
    return compacted


def _compact_job_for_state(job: dict[str, Any]) -> dict[str, Any]:
    next_job = {
        "title": job.get("title"),
        "company": job.get("company"),
        "location": job.get("location"),
        "canonical_url": job.get("canonical_url") or job.get("final_url") or job.get("url"),
        "source_urls": job.get("source_urls"),
        "dedupe_key": job.get("dedupe_key") or job.get("duplicate_key"),
        "duplicate_key": job.get("duplicate_key") or job.get("dedupe_key"),
        "verification_status": job.get("verification_status"),
        "status": job.get("status"),
    }
    next_job["source_urls"] = _compact_string_list(next_job.get("source_urls"), limit=MAX_STATE_SOURCE_URLS, max_chars=500)
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

    return [_job_dedupe_record(job) for job in merged.values()]


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
    duplicate_key = str(job.get("duplicate_key") or job.get("dedupe_key") or "").strip()
    if _has_strong_dedupe_key(duplicate_key):
        for candidate_key, candidate in merged.items():
            if str(candidate.get("duplicate_key") or candidate.get("dedupe_key") or "").strip() == duplicate_key:
                return candidate_key
    return job_id


def _merge_job(existing: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    first_seen = existing.get("first_seen_at") or job.get("first_seen_at") or _today()
    canonical_url_changed = _canonical_url_changed(existing, job)
    source_urls = _merge_unique_urls(existing.get("source_urls", []), job.get("source_urls", []))
    if job.get("canonical_url"):
        source_urls = _merge_unique_urls(source_urls, [job["canonical_url"]])

    next_job = _merge_job_fields(existing, job)
    next_job["job_id"] = existing.get("job_id") or job["job_id"]
    next_job["first_seen_at"] = first_seen
    next_job["last_seen_at"] = job.get("last_seen_at") or _today()
    next_job["source_urls"] = source_urls
    explicit_status = _status_from_explicit_verification(job)
    if explicit_status is not None:
        if explicit_status == "unknown" and existing.get("status") == "closed" and not canonical_url_changed:
            next_job["status"] = "closed"
        else:
            next_job["status"] = explicit_status
    elif _should_reactivate_closed_status(existing, job, canonical_url_changed):
        next_job["status"] = "active"
    else:
        next_job.setdefault("status", "active")
    return next_job


def _job_dedupe_record(job: dict[str, Any]) -> dict[str, Any]:
    source_urls = _compact_string_list(job.get("source_urls"), limit=MAX_STATE_SOURCE_URLS, max_chars=500)
    status = str(job.get("status") or "").strip().lower()
    dedupe_key = str(job.get("dedupe_key") or job.get("duplicate_key") or "").strip()
    record = {
        "title": str(job.get("title") or "").strip(),
        "company": str(job.get("company") or "").strip(),
        "location": str(job.get("location") or "").strip(),
        "canonical_url": str(job.get("canonical_url") or "").strip(),
        "source_urls": source_urls,
        "dedupe_key": dedupe_key if _has_strong_dedupe_key(dedupe_key) else "",
        "first_seen_at": str(job.get("first_seen_at") or "").strip(),
        "last_seen_at": str(job.get("last_seen_at") or "").strip(),
    }
    if status and status != "active":
        record["status"] = status
    return {
        key: value
        for key, value in record.items()
        if (key == "status" and value in {"closed", "unknown"}) or _has_meaningful_value(value)
    }


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


def _canonical_url_changed(existing: dict[str, Any], job: dict[str, Any]) -> bool:
    incoming_url = _canonical_url_key(job.get("canonical_url"))
    if not incoming_url:
        return False
    existing_url = _canonical_url_key(existing.get("canonical_url"))
    return incoming_url != existing_url


def _should_reactivate_closed_status(
    existing: dict[str, Any],
    job: dict[str, Any],
    canonical_url_changed: bool,
) -> bool:
    return (
        str(existing.get("status") or "").strip().lower() == "closed"
        and str(job.get("status") or "").strip().lower() == "active"
        and canonical_url_changed
    )


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
    next_job["canonical_url"] = _normalize_canonical_url(next_job.get("canonical_url"))
    next_job["source_urls"] = _normalize_url_list(next_job.get("source_urls"))
    next_job["normalized_company"] = _slug(next_job.get("normalized_company") or next_job["company"])
    next_job["normalized_title"] = _slug(next_job.get("normalized_title") or next_job["title"])
    next_job["normalized_location"] = _slug(next_job.get("normalized_location") or next_job["location"])
    field_dedupe_key = _dedupe_key_from_fields(next_job)
    input_dedupe_key = _normalize_dedupe_key(next_job.get("dedupe_key") or next_job.get("duplicate_key"))
    dedupe_key = field_dedupe_key if _has_strong_dedupe_key(field_dedupe_key) else input_dedupe_key or field_dedupe_key
    next_job["dedupe_key"] = dedupe_key
    next_job["duplicate_key"] = dedupe_key
    next_job["job_id"] = next_job.get("job_id") or _job_id(next_job)
    next_job.setdefault("first_seen_at", _today())
    next_job.setdefault("last_seen_at", _today())
    next_job.setdefault("status", _status_from_verification(next_job))
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
    return _normalize_canonical_url(value)


def _slug(value: Any) -> str:
    if _is_placeholder_text(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value or "").lower().strip())
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def _dedupe_key_from_fields(job: dict[str, Any]) -> str:
    return "|".join(
        [
            _slug(job.get("company")),
            _slug(job.get("title")),
            _slug(job.get("location")),
        ]
    )


def _normalize_dedupe_key(value: Any) -> str:
    text = str(value or "").strip()
    if not text or _is_placeholder_text(text):
        return ""
    if "|" in text:
        return "|".join(_slug(part) for part in text.split("|"))
    return _slug(text)


def _has_strong_dedupe_key(value: str) -> bool:
    parts = value.split("|")
    return len(parts) == 3 and all(bool(part.strip()) for part in parts)


def _normalize_canonical_url(value: Any) -> str:
    text = str(value or "").strip()
    if not text or _is_placeholder_text(text):
        return ""

    try:
        parsed = urlsplit(text)
    except ValueError:
        return text.rstrip("/")

    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return text.rstrip("/")

    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").strip("[]").rstrip(".").lower()
    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError:
        pass
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"

    port = _url_port(parsed)
    default_port = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    netloc = host if port is None or default_port else f"{host}:{port}"

    path = parsed.path or ""
    if path and path != "/":
        path = re.sub(r"/{2,}", "/", path).rstrip("/")
    else:
        path = ""

    query_items = []
    for key, query_value in parse_qsl(parsed.query, keep_blank_values=True):
        normalized_key = key.lower()
        if normalized_key in TRACKING_QUERY_PARAMS:
            continue
        if any(normalized_key.startswith(prefix) for prefix in TRACKING_QUERY_PREFIXES):
            continue
        query_items.append((key, query_value))
    query_items.sort(key=lambda item: (item[0].lower(), item[1]))
    query = urlencode(query_items, doseq=True)

    return urlunsplit((scheme, netloc, path, query, ""))


def _url_port(parsed: Any) -> int | None:
    try:
        return parsed.port
    except ValueError:
        return None


def _normalize_url_list(value: Any) -> list[str]:
    values: list[str] = []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        items = []
    for item in items:
        normalized = _normalize_canonical_url(item)
        key = _canonical_url_key(normalized)
        if normalized and key and key not in {_canonical_url_key(existing) for existing in values}:
            values.append(normalized)
    return values


def _merge_unique_urls(first: Any, second: Any) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for source in (first, second):
        if isinstance(source, str):
            items = [source]
        elif isinstance(source, list):
            items = source
        else:
            items = []
        for item in items:
            text = _normalize_canonical_url(item)
            key = _canonical_url_key(text)
            if text and key and key not in seen:
                values.append(text)
                seen.add(key)
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


def _select_dedupe_jobs(jobs: list[Any], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    indexed_jobs = [(index, job) for index, job in enumerate(jobs) if isinstance(job, dict)]
    selected = sorted(indexed_jobs, key=_dedupe_selection_key)[:limit]
    return [job for _index, job in selected]


def _dedupe_selection_key(indexed_job: tuple[int, dict[str, Any]]) -> tuple[int, float, int]:
    index, job = indexed_job
    status = _dedupe_status(job)
    return (
        DEDUPE_STATUS_RANK.get(status, DEDUPE_STATUS_RANK["active"]),
        -_dedupe_seen_timestamp(job),
        -index,
    )


def _dedupe_status(job: dict[str, Any]) -> str:
    status = str(job.get("status") or "").strip().lower()
    return status if status in DEDUPE_STATUS_RANK else "active"


def _dedupe_seen_timestamp(job: dict[str, Any]) -> float:
    for key in ("last_seen_at", "updated_at", "first_seen_at"):
        timestamp = _parse_datetime_timestamp(job.get(key))
        if timestamp:
            return timestamp
    return 0.0


def _parse_datetime_timestamp(value: Any) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return 0.0


def _dedupe_brief(jobs: list[dict[str, Any]]) -> str:
    if not jobs:
        return "No previous job dedupe records."
    lines = ["Known previous job dedupe records:"]
    for job in jobs:
        title = job.get("title") or "Unknown title"
        company = job.get("company") or "Unknown company"
        location = job.get("location") or "Unknown location"
        dedupe_key = job.get("dedupe_key") or ""
        canonical_url = job.get("canonical_url") or ""
        status = job.get("status") or "active_or_previous"
        last_seen = job.get("last_seen_at") or "unknown"
        lines.append(
            f"- {company} | {title} | {location} | status={status} | last_seen={last_seen} | dedupe_key={dedupe_key} | url={canonical_url}"
        )
    return "\n".join(lines)


WORKSPACE_TOOLS = [
    start_workspace_run,
    save_workspace_file,
    save_job_artifact,
    read_job_search_dedupe_state,
    update_job_search_state,
    update_job_search_state_from_artifact,
]
