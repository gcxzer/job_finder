from __future__ import annotations

import ast
import hashlib
import json
import re
import shlex
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from langchain_core.tools import tool

from src.configs import CONFIG
from src.tools.url_safety import public_http_url_error

WORKSPACE_DIR = CONFIG.workspace.root_dir
CRAWLER_ROOT = CONFIG.workspace.crawlers_dir
CONTAINER_WORKSPACE = CONFIG.docker.container_workspace_dir
MAX_PREVIEW_CHARS = 1600
CRAWLER_SCHEMA_VERSION = "job_extraction_context_v1"
CRAWLER_CODE_FILE_NAME = "crawler.py"
CRAWLER_RESULT_FILE_NAME = "job_result.json"
CRAWLER_LOG_FILE_NAME = "crawler.log"
CRAWLER_RUNTIME_GUARD_FILE_NAME = "sitecustomize.py"
CRAWLER_SETUP_TIMEOUT_SECONDS = 120
CRAWLER_DEPENDENCY_PACKAGES = {
    "requests": "requests==2.34.2",
    "bs4": "beautifulsoup4==4.14.3",
    "lxml": "lxml==6.1.1",
}
CREDENTIAL_ENV_NAME_PATTERN = re.compile(
    r"(?:^|_)(?:api_?key|auth|bearer|cookie|credential|login|pass(?:word)?|secret|session|token|user(?:name)?)(?:_|$)",
    re.I,
)
CREDENTIAL_HEADER_NAMES = {
    "authorization",
    "cookie",
    "proxy-authorization",
    "x-api-key",
}
ALLOWED_CRAWLER_ENV_NAMES = {"TARGET_URL", "OUTPUT_FILE"}
FORBIDDEN_IMPORT_ROOTS = {
    "aiohttp",
    "builtins",
    "ctypes",
    "ftplib",
    "glob",
    "httpx",
    "importlib",
    "multiprocessing",
    "shutil",
    "sitecustomize",
    "socket",
    "socketserver",
    "subprocess",
    "sys",
}
FORBIDDEN_BUILTIN_CALLS = {
    "__import__",
    "compile",
    "delattr",
    "eval",
    "exec",
    "getattr",
    "globals",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}
FORBIDDEN_OS_CALLS = {
    "listdir",
    "popen",
    "remove",
    "removedirs",
    "rename",
    "replace",
    "rmdir",
    "scandir",
    "spawnl",
    "spawnle",
    "spawnlp",
    "spawnlpe",
    "spawnv",
    "spawnve",
    "spawnvp",
    "spawnvpe",
    "startfile",
    "system",
    "unlink",
    "walk",
}
HTTP_REQUEST_METHODS = {"delete", "get", "head", "options", "patch", "post", "put", "request"}
SECTION_HINT_PATTERN = re.compile(
    r"job|career|position|posting|description|requirement|qualification|benefit|apply|bewerb|stelle|aufgabe|profil",
    re.I,
)
CLOUDFLARE_CHALLENGE_PATTERN = re.compile(
    r"cf-chl|__cf_chl_|/cdn-cgi/challenge-platform|cf-browser-verification",
    re.I,
)
REQUIRED_CRAWLER_CONTEXT_KEYS = {
    "success",
    "schema_version",
    "url",
    "final_url",
    "extraction_method",
    "technical_status",
    "verification_status",
    "standard_extraction",
    "page_context",
    "technical_signals",
    "verified_at",
    "error",
}


# Public tool entry points


@tool
def analyze_job_html_structure(url: str = "", html: str = "", html_file: str = "") -> dict[str, Any]:
    """Analyze cached HTML and return compact DOM hints for generating a job crawler."""
    content = _load_html(html=html, html_file=html_file)
    if not content:
        return {"success": False, "error": "No HTML content provided.", "url": url}

    soup = BeautifulSoup(content, "lxml")
    text = _visible_text(soup)
    tags = [tag.name for tag in soup.find_all()]
    tag_counter = Counter(tags)

    anti_scraping = _detect_anti_scraping(content, text)
    json_ld_types = _json_ld_types(soup)
    likely_job_sections = _likely_job_sections(soup)

    return {
        "success": True,
        "url": url,
        "title": _collapse_text(soup.title.string if soup.title else ""),
        "text_length": len(text),
        "text_preview": text[:MAX_PREVIEW_CHARS],
        "total_tags": len(tags),
        "tag_distribution": dict(tag_counter.most_common(12)),
        "json_ld_types": json_ld_types[:12],
        "common_containers": _common_containers(soup)[:16],
        "sample_links": _sample_links(soup, url)[:20],
        "likely_job_sections": likely_job_sections[:12],
        "anti_scraping": anti_scraping,
        "recommended_strategy": _recommended_strategy(anti_scraping, json_ld_types, text),
    }


@tool
def save_job_crawler_code(code: str, run_id: str = "ad_hoc", job_id: str = "job") -> dict[str, Any]:
    """Save generated crawler code under workspace/crawlers or a run snapshot."""
    clean_code = _strip_code_fence(code)
    rel_dir = _crawler_dir(run_id=run_id, job_id=job_id)
    rel_dir.mkdir(parents=True, exist_ok=True)
    code_path = rel_dir / CRAWLER_CODE_FILE_NAME
    code_path.write_text(clean_code, encoding="utf-8")
    output_path = rel_dir / CRAWLER_RESULT_FILE_NAME
    log_path = rel_dir / CRAWLER_LOG_FILE_NAME
    return {
        "success": True,
        "code_file": str(code_path),
        "container_code_file": _container_path(code_path),
        "result_file": str(output_path),
        "container_result_file": _container_path(output_path),
        "log_file": str(log_path),
        "container_log_file": _container_path(log_path),
        "bytes": len(clean_code.encode("utf-8")),
    }


@tool
def validate_job_crawler_code(code_file: str) -> dict[str, Any]:
    """Validate generated crawler Python syntax and the required output contract."""
    path, error = _existing_crawler_code_file_or_error(code_file)
    if error is not None:
        return error
    assert path is not None

    parse_error, contract_errors = _crawler_validation_errors(path)
    validation_error = _crawler_validation_error_response(
        path,
        parse_error,
        contract_errors,
        include_container_for_parse_error=False,
    )
    if validation_error is not None:
        return validation_error

    return {
        "success": True,
        "code_file": str(path),
        "container_code_file": _container_path(path),
        "validated_code_sha256": _file_sha256(path),
    }


@tool
def build_job_crawler_run_command(code_file: str, url: str, timeout: int = 60) -> dict[str, Any]:
    """Build commands for running crawler.py through the DeepAgents execute tool."""
    url_error = public_http_url_error(url)
    if url_error:
        return {"success": False, "error": f"Unsafe crawler URL: {url_error}", "url": url}

    path, error = _existing_crawler_code_file_or_error(code_file)
    if error is not None:
        return error
    assert path is not None

    parse_error, contract_errors = _crawler_validation_errors(path)
    validation_error = _crawler_validation_error_response(
        path,
        parse_error,
        contract_errors,
        include_container_for_parse_error=True,
    )
    if validation_error is not None:
        return validation_error

    run_dir = path.parent
    output_path = run_dir / CRAWLER_RESULT_FILE_NAME
    log_path = run_dir / CRAWLER_LOG_FILE_NAME
    guard_path = run_dir / CRAWLER_RUNTIME_GUARD_FILE_NAME
    guard_path.write_text(_crawler_runtime_guard_source(), encoding="utf-8")
    container_code_file = _container_path(path)
    container_output_file = _container_path(output_path)
    container_log_file = _container_path(log_path)
    container_run_dir = _container_path(run_dir)
    container_guard_file = _container_path(guard_path)
    return {
        "success": True,
        "code_file": str(path),
        "container_code_file": container_code_file,
        "result_file": str(output_path),
        "container_result_file": container_output_file,
        "log_file": str(log_path),
        "container_log_file": container_log_file,
        "runtime_guard_file": str(guard_path),
        "container_runtime_guard_file": container_guard_file,
        "validated_code_sha256": _file_sha256(path),
        "setup_command": _crawler_dependency_setup_command(),
        "run_command": _crawler_run_command(
            url=url,
            timeout=timeout,
            container_code_file=container_code_file,
            container_output_file=container_output_file,
            container_log_file=container_log_file,
            container_run_dir=container_run_dir,
        ),
        "read_result_instruction": f"After execute succeeds, read {container_output_file}.",
    }


# Workspace path and command helpers


def _crawler_dir(run_id: str, job_id: str) -> Path:
    safe_run_id = _safe_name(run_id or "ad_hoc")
    safe_job_id = _safe_name(job_id or "job")
    if safe_run_id == "ad_hoc":
        safe_run_id = datetime.now().astimezone().strftime("%Y-%m-%d_%H%M%S")
        return CRAWLER_ROOT / safe_run_id / safe_job_id
    return WORKSPACE_DIR / "runs" / safe_run_id / "crawlers" / safe_job_id


def _resolve_workspace_file(file_path: str) -> Path | None:
    clean_path = str(file_path or "").strip()
    if not clean_path:
        return None

    container_workspace = CONTAINER_WORKSPACE.rstrip("/")
    if clean_path == container_workspace:
        path = WORKSPACE_DIR
    elif clean_path.startswith(f"{container_workspace}/"):
        path = WORKSPACE_DIR / clean_path[len(container_workspace) :].lstrip("/")
    else:
        raw_path = Path(clean_path).expanduser()
        path = raw_path if raw_path.is_absolute() else WORKSPACE_DIR / clean_path.lstrip("/")

    resolved = path.resolve()
    try:
        resolved.relative_to(WORKSPACE_DIR)
    except ValueError:
        return None
    return resolved


def _existing_crawler_code_file_or_error(code_file: str) -> tuple[Path | None, dict[str, Any] | None]:
    path = _resolve_workspace_file(code_file)
    if path is None or not path.exists():
        return None, {"success": False, "error": "Crawler code file not found.", "code_file": code_file}
    return path, None


def _container_path(path: Path) -> str:
    rel = path.resolve().relative_to(WORKSPACE_DIR)
    return f"{CONTAINER_WORKSPACE}/{rel.as_posix()}"


def _load_html(*, html: str, html_file: str) -> str:
    if html:
        return html
    if html_file:
        path = _resolve_workspace_file(html_file)
        if path is not None and path.exists() and path.is_file():
            return path.read_text(encoding="utf-8", errors="ignore")
    return ""


def _crawler_run_command(
    *,
    url: str,
    timeout: int,
    container_code_file: str,
    container_output_file: str,
    container_log_file: str,
    container_run_dir: str,
) -> str:
    return " ".join(
        [
            "set -o pipefail;",
            f"rm -f {shlex.quote(container_output_file)} {shlex.quote(container_log_file)} &&",
            f"timeout {int(timeout)}s",
            "env",
            f"TARGET_URL={shlex.quote(url)}",
            f"OUTPUT_FILE={shlex.quote(container_output_file)}",
            f"PYTHONPATH={shlex.quote(container_run_dir)}",
            f"python {shlex.quote(container_code_file)}",
            f"2>&1 | tee {shlex.quote(container_log_file)}",
        ]
    )


# Crawler dependency setup and validation entry helpers


def _crawler_dependency_setup_command() -> str:
    packages_json = json.dumps(CRAWLER_DEPENDENCY_PACKAGES, sort_keys=True)
    return (
        f"timeout {CRAWLER_SETUP_TIMEOUT_SECONDS}s python - <<'PY'\n"
        "import importlib.util\n"
        "import os\n"
        "import subprocess\n"
        "import sys\n"
        f"packages = {packages_json}\n"
        "missing = [package for module, package in packages.items() if importlib.util.find_spec(module) is None]\n"
        "if missing:\n"
        "    env = dict(os.environ)\n"
        "    env.setdefault('PIP_DISABLE_PIP_VERSION_CHECK', '1')\n"
        "    env.setdefault('PIP_NO_INPUT', '1')\n"
        "    env.setdefault('PIP_DEFAULT_TIMEOUT', '30')\n"
        "    print('installing crawler dependencies: ' + ', '.join(missing), flush=True)\n"
        "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', *missing], env=env)\n"
        "else:\n"
        "    print('crawler dependencies already installed')\n"
        "PY"
    )


def _crawler_validation_errors(path: Path) -> tuple[str | None, list[str]]:
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return f"SyntaxError line {error.lineno}: {error.msg}", []
    return None, _crawler_contract_errors(source, tree)


def _crawler_validation_error_response(
    path: Path,
    parse_error: str | None,
    contract_errors: list[str],
    *,
    include_container_for_parse_error: bool,
) -> dict[str, Any] | None:
    if parse_error is not None:
        response: dict[str, Any] = {
            "success": False,
            "code_file": str(path),
            "error": parse_error,
        }
        if include_container_for_parse_error:
            response["container_code_file"] = _container_path(path)
        return response

    if contract_errors:
        return {
            "success": False,
            "code_file": str(path),
            "container_code_file": _container_path(path),
            "error": "Crawler contract validation failed.",
            "contract_errors": contract_errors,
        }

    return None


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Runtime guard source written next to generated crawlers


def _crawler_runtime_guard_source() -> str:
    return r'''
"""Runtime guard for generated job crawler code."""
from __future__ import annotations

import builtins
import ipaddress
import os
import shutil
import socket
import subprocess
from urllib.parse import urlparse

_FORBIDDEN_IMPORT_ROOTS = {
    "aiohttp",
    "builtins",
    "ctypes",
    "ftplib",
    "httpx",
    "importlib",
    "multiprocessing",
    "socketserver",
    "subprocess",
    "sys",
    "sitecustomize",
}

_TARGET_URL = os.environ.get("TARGET_URL", "")
_TARGET = urlparse(_TARGET_URL)
_TARGET_HOST = (_TARGET.hostname or "").strip("[]").rstrip(".").lower()
if not _TARGET_HOST:
    raise RuntimeError("TARGET_URL must include a hostname.")


def _blocked_process(*args: object, **kwargs: object) -> None:
    raise RuntimeError("Generated crawler code is not allowed to spawn processes or run shell commands.")


def _install_guard() -> None:
    original_import = builtins.__import__
    original_socket_connect = socket.socket.connect
    original_create_connection = socket.create_connection
    original_getaddrinfo = socket.getaddrinfo

    def is_ip(value: str) -> bool:
        try:
            ipaddress.ip_address(value)
        except ValueError:
            return False
        return True

    def blocked_ip(value: str) -> bool:
        try:
            address = ipaddress.ip_address(value)
        except ValueError:
            return False
        return bool(
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_multicast
            or address.is_reserved
            or address.is_unspecified
        )

    def resolved_ips(host: str) -> set[str]:
        try:
            addresses = original_getaddrinfo(host, None, type=socket.SOCK_STREAM)
        except socket.gaierror as error:
            raise RuntimeError(f"Blocked crawler connection: hostname could not be resolved: {error}.") from error
        return {str(address[4][0]) for address in addresses}

    def clean_host(host: object) -> str:
        return str(host or "").strip("[]").rstrip(".").lower()

    target_ips = {_TARGET_HOST} if is_ip(_TARGET_HOST) else resolved_ips(_TARGET_HOST)
    for ip_value in target_ips:
        if blocked_ip(ip_value):
            raise RuntimeError("TARGET_URL resolves to a private, local, link-local, reserved, multicast, or unspecified IP.")

    def check_host(host: object) -> None:
        checked_host = clean_host(host)
        if not checked_host:
            raise RuntimeError("Blocked crawler connection without a hostname.")
        if checked_host != _TARGET_HOST and checked_host not in target_ips:
            raise RuntimeError(f"Blocked crawler connection to {checked_host}; only TARGET_URL host {_TARGET_HOST} is allowed.")
        if blocked_ip(checked_host):
            raise RuntimeError("Blocked crawler connection to a private, local, link-local, reserved, multicast, or unspecified IP.")
        if not is_ip(checked_host):
            for ip_value in resolved_ips(checked_host):
                if blocked_ip(ip_value):
                    raise RuntimeError(
                        "Blocked crawler connection to a hostname resolving to a private, local, link-local, reserved, "
                        "multicast, or unspecified IP."
                    )

    def guarded_socket_connect(self: socket.socket, address: object) -> object:
        host = address[0] if isinstance(address, tuple) and address else address
        check_host(host)
        return original_socket_connect(self, address)

    def guarded_create_connection(address: object, timeout: object = socket._GLOBAL_DEFAULT_TIMEOUT, source_address: object = None, *args: object, **kwargs: object) -> object:
        host = address[0] if isinstance(address, tuple) and address else address
        check_host(host)
        return original_create_connection(address, timeout, source_address, *args, **kwargs)

    def guarded_import(name: str, globals: object = None, locals: object = None, fromlist: object = (), level: int = 0) -> object:
        root = str(name or "").split(".", 1)[0]
        if root in _FORBIDDEN_IMPORT_ROOTS:
            raise ImportError(f"Generated crawler code is not allowed to import {root}.")
        return original_import(name, globals, locals, fromlist, level)

    socket.socket.connect = guarded_socket_connect
    socket.create_connection = guarded_create_connection
    builtins.__import__ = guarded_import


_install_guard()
del _install_guard
for _name in ("Popen", "run", "call", "check_call", "check_output"):
    setattr(subprocess, _name, _blocked_process)
for _name in ("popen", "spawnl", "spawnle", "spawnlp", "spawnlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe", "system"):
    if hasattr(os, _name):
        setattr(os, _name, _blocked_process)
for _name in ("copy", "copy2", "copyfile", "copytree", "move", "rmtree"):
    if hasattr(shutil, _name):
        setattr(shutil, _name, _blocked_process)
'''


# Crawler contract validation


def _crawler_contract_errors(source: str, tree: ast.AST) -> list[str]:
    constants = _string_constants(tree)
    errors: list[str] = []

    for env_name in ("TARGET_URL", "OUTPUT_FILE"):
        if not _reads_env_name(tree, env_name):
            errors.append(f"Must read {env_name} from the environment.")

    if CRAWLER_SCHEMA_VERSION not in constants:
        errors.append(f'Must emit schema_version "{CRAWLER_SCHEMA_VERSION}".')

    missing_keys = sorted(REQUIRED_CRAWLER_CONTEXT_KEYS - constants)
    if missing_keys:
        errors.append(f"Must include context keys: {', '.join(missing_keys)}.")

    if not _has_json_serialization(tree):
        errors.append("Must serialize the result with json.dump/json.dumps.")

    if not _writes_to_output_file(tree):
        errors.append("Must write the JSON result to OUTPUT_FILE.")

    if not _uses_requests_session(tree):
        errors.append("Must fetch pages with requests.Session.")

    if not _uses_beautifulsoup(tree):
        errors.append("Must parse HTML with BeautifulSoup/lxml.")

    if _imports_rendering_browser_tool(tree):
        errors.append("Generated crawler must not use Playwright or Selenium. Use browser_extract_job_page before generating crawler code.")

    errors.extend(_dangerous_code_errors(tree))
    errors.extend(_file_access_errors(tree))
    errors.extend(_network_target_errors(tree))
    errors.extend(_credential_usage_errors(tree))

    if "print(" in source and not _writes_to_output_file(tree):
        errors.append("Printing alone is not enough; write a JSON object to OUTPUT_FILE.")

    return errors


def _dangerous_code_errors(tree: ast.AST) -> list[str]:
    errors: list[str] = []
    forbidden_imports = sorted(_forbidden_import_roots(tree))
    if forbidden_imports:
        errors.append("Must not import high-risk modules: " + ", ".join(forbidden_imports) + ".")

    dangerous_calls = sorted(_dangerous_call_names(tree))
    if dangerous_calls:
        errors.append("Must not use dynamic execution or process/file-system helpers: " + ", ".join(dangerous_calls) + ".")

    sensitive_assignments = sorted(_sensitive_assignment_names(tree))
    if sensitive_assignments:
        errors.append("Must not modify runtime guard or networking internals: " + ", ".join(sensitive_assignments) + ".")

    return errors


def _forbidden_import_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name
                root = module.split(".", 1)[0]
                if root in FORBIDDEN_IMPORT_ROOTS:
                    roots.add(root)
                elif module == "urllib" or module.startswith("urllib.") and module != "urllib.parse":
                    roots.add(module)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".", 1)[0]
            if root in FORBIDDEN_IMPORT_ROOTS:
                roots.add(root)
            elif module == "urllib" or module.startswith("urllib.") and module != "urllib.parse":
                roots.add(module)
    return roots


def _dangerous_call_names(tree: ast.AST) -> set[str]:
    calls: set[str] = set()
    output_references = _output_file_references(tree)
    path_constructors = _path_constructor_refs(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in FORBIDDEN_BUILTIN_CALLS:
            if func.id == "open" and _is_output_open_call(node, output_references, path_constructors):
                continue
            calls.add(func.id)
        elif isinstance(func, ast.Attribute):
            if isinstance(func.value, ast.Name) and func.value.id == "os" and func.attr in FORBIDDEN_OS_CALLS:
                calls.add(f"os.{func.attr}")
    return calls


def _sensitive_assignment_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        elif isinstance(node, ast.AugAssign):
            targets = [node.target]
        for target in targets:
            path = _attribute_path(target)
            if not path:
                continue
            if (
                path == "builtins.__import__"
                or path == "socket.create_connection"
                or path == "socket.socket.connect"
                or path.startswith("sitecustomize.")
            ):
                names.add(path)
    return names


def _attribute_path(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _attribute_path(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _file_access_errors(tree: ast.AST) -> list[str]:
    references = _output_file_references(tree)
    path_constructors = _path_constructor_refs(tree)
    errors: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "open":
            if _is_output_open_call(node, references, path_constructors):
                continue
            errors.append("Must not open files other than OUTPUT_FILE.")
        elif isinstance(func, ast.Attribute):
            if func.attr == "open":
                if _is_output_open_call(node, references, path_constructors):
                    continue
                errors.append("Must not open files other than OUTPUT_FILE.")
            if func.attr in {"read_bytes", "read_text"}:
                errors.append("Must not read local files from generated crawler code.")
            elif func.attr in {"touch", "unlink", "write_bytes", "write_text"} and not _is_exact_output_path_reference(
                func.value,
                references,
                path_constructors,
            ):
                errors.append("Must write only to OUTPUT_FILE.")
    return _dedupe_errors(errors)


def _network_target_errors(tree: ast.AST) -> list[str]:
    target_aliases = _direct_target_url_aliases(tree)
    session_aliases = _requests_session_aliases(tree)
    errors: list[str] = []

    if _literal_http_urls(tree):
        errors.append("Must not hard-code HTTP URLs; read and fetch TARGET_URL only.")

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        url_arg = _request_url_arg(node, session_aliases)
        if url_arg is None:
            continue
        if not _is_exact_target_url_reference(url_arg, target_aliases):
            errors.append("HTTP requests in generated crawler code must use TARGET_URL directly.")

    return _dedupe_errors(errors)


def _direct_target_url_aliases(tree: ast.AST) -> set[str]:
    aliases = _env_value_aliases(tree, "TARGET_URL")
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            targets: list[ast.expr] = []
            value: ast.AST | None = None
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
                value = node.value
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
                value = node.value
            if value is None:
                continue
            if _is_exact_target_url_reference(value, aliases):
                before = len(aliases)
                aliases.update(_assigned_names(targets))
                changed = changed or len(aliases) > before
    return aliases


def _literal_http_urls(tree: ast.AST) -> set[str]:
    urls: set[str] = set()
    for value in _string_constants(tree):
        if re.match(r"https?://", value.strip(), flags=re.I):
            urls.add(value)
    return urls


def _request_url_arg(node: ast.Call, session_aliases: set[str]) -> ast.AST | None:
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr not in HTTP_REQUEST_METHODS:
        return None

    if not _is_http_request_receiver(func.value, session_aliases):
        return None
    return _request_call_url_argument(node, func.attr)


def _is_http_request_receiver(node: ast.AST, session_aliases: set[str]) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "requests" or node.id in session_aliases
    return isinstance(node, ast.Call) and _is_requests_session_call(node)


def _request_call_url_argument(node: ast.Call, method_name: str) -> ast.AST | None:
    if method_name == "request":
        return node.args[1] if len(node.args) >= 2 else _keyword_value(node, "url")
    return node.args[0] if node.args else _keyword_value(node, "url")


def _requests_session_aliases(tree: ast.AST) -> set[str]:
    aliases: set[str] = set()
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        value: ast.AST | None = None
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        if value is not None and _is_requests_session_call(value):
            aliases.update(_assigned_names(targets))
    return aliases


def _is_requests_session_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "Session"
        and isinstance(func.value, ast.Name)
        and func.value.id == "requests"
    )


def _is_exact_target_url_reference(node: ast.AST, target_aliases: set[str]) -> bool:
    if isinstance(node, ast.Name) and node.id in target_aliases:
        return True
    return _node_reads_env_name(node, "TARGET_URL")


def _keyword_value(node: ast.Call, name: str) -> ast.AST | None:
    for keyword in node.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _output_file_references(tree: ast.AST) -> set[str]:
    output_aliases = _env_value_aliases(tree, "OUTPUT_FILE")
    path_constructors = _path_constructor_refs(tree)
    return output_aliases | _path_aliases_from_output(tree, output_aliases, path_constructors)


def _dedupe_errors(errors: list[str]) -> list[str]:
    deduped: list[str] = []
    for error in errors:
        if error not in deduped:
            deduped.append(error)
    return deduped


def _credential_usage_errors(tree: ast.AST) -> list[str]:
    errors: list[str] = []
    credential_env_names = sorted(_credential_env_names(tree))
    if credential_env_names:
        errors.append(
            "Must not read credential-like environment variables: "
            + ", ".join(credential_env_names)
            + ".",
        )

    if _uses_auth_argument_or_attribute(tree):
        errors.append("Must not pass credentials with auth= or session.auth.")

    credential_headers = sorted(_credential_header_names(tree))
    if credential_headers:
        errors.append("Must not send credential headers: " + ", ".join(credential_headers) + ".")

    return errors


def _string_constants(tree: ast.AST) -> set[str]:
    values: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            values.add(node.value)
    return values


def _has_json_serialization(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in {"dump", "dumps"}:
            if isinstance(func.value, ast.Name) and func.value.id == "json":
                return True
        if isinstance(func, ast.Name) and func.id in {"dump", "dumps"}:
            return True
    return False


def _reads_env_name(tree: ast.AST, env_name: str) -> bool:
    return any(_node_reads_env_name(node, env_name) for node in ast.walk(tree))


def _node_reads_env_name(node: ast.AST, env_name: str) -> bool:
    return _env_name_read_by_node(node) == env_name


def _env_name_read_by_node(node: ast.AST) -> str | None:
    if isinstance(node, ast.Subscript) and _is_os_environ(node.value):
        return _slice_string(node.slice)

    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "get" and _is_os_environ(func.value):
            return _literal_string(node.args[0]) if node.args else None
        if isinstance(func, ast.Attribute) and func.attr == "getenv":
            if isinstance(func.value, ast.Name) and func.value.id == "os":
                return _literal_string(node.args[0]) if node.args else None
        if isinstance(func, ast.Name) and func.id == "getenv":
            return _literal_string(node.args[0]) if node.args else None

    return None


def _credential_env_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        env_name = _env_name_read_by_node(node)
        if not env_name or env_name in ALLOWED_CRAWLER_ENV_NAMES:
            continue
        if CREDENTIAL_ENV_NAME_PATTERN.search(env_name):
            names.add(env_name)
    return names


def _uses_auth_argument_or_attribute(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if any(keyword.arg == "auth" for keyword in node.keywords):
                return True
            func = node.func
            if isinstance(func, ast.Name) and func.id in {"HTTPBasicAuth", "HTTPDigestAuth"}:
                return True
            if isinstance(func, ast.Attribute) and func.attr in {"HTTPBasicAuth", "HTTPDigestAuth"}:
                return True
        elif isinstance(node, ast.Assign):
            if any(_is_auth_attribute(target) for target in node.targets):
                return True
        elif isinstance(node, ast.AnnAssign):
            if _is_auth_attribute(node.target):
                return True
        elif isinstance(node, ast.AugAssign):
            if _is_auth_attribute(node.target):
                return True
    return False


def _is_auth_attribute(node: ast.AST) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "auth"


def _credential_header_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key in node.keys:
                header = _literal_string(key)
                if header and header.lower() in CREDENTIAL_HEADER_NAMES:
                    names.add(header)
        elif isinstance(node, ast.Subscript):
            header = _slice_string(node.slice)
            if header and header.lower() in CREDENTIAL_HEADER_NAMES:
                names.add(header)
    return names


def _is_os_environ(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "environ"
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
    )


def _slice_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _literal_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _writes_to_output_file(tree: ast.AST) -> bool:
    output_aliases = _env_value_aliases(tree, "OUTPUT_FILE")
    path_constructors = _path_constructor_refs(tree)
    path_aliases = _path_aliases_from_output(tree, output_aliases, path_constructors)
    references = output_aliases | path_aliases
    writer_aliases = _writer_aliases_for_output(tree, references, path_constructors)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "write_text":
            if _is_exact_output_path_reference(func.value, references, path_constructors):
                return True
        if _call_writes_to_output_handle(node, writer_aliases, references, path_constructors):
            return True
    return False


def _writer_aliases_for_output(
    tree: ast.AST,
    references: set[str],
    path_constructors: tuple[set[str], set[str]],
) -> set[str]:
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.With):
            for item in node.items:
                if item.optional_vars is not None and _is_output_open_call(
                    item.context_expr,
                    references,
                    path_constructors,
                ):
                    aliases.update(_assigned_names([item.optional_vars]))
        elif isinstance(node, ast.Assign):
            if _is_output_open_call(node.value, references, path_constructors):
                aliases.update(_assigned_names(list(node.targets)))
        elif isinstance(node, ast.AnnAssign):
            if node.value is not None and _is_output_open_call(
                node.value,
                references,
                path_constructors,
            ):
                aliases.update(_assigned_names([node.target]))
    return aliases


def _call_writes_to_output_handle(
    node: ast.Call,
    writer_aliases: set[str],
    references: set[str],
    path_constructors: tuple[set[str], set[str]],
) -> bool:
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr in {"write", "writelines"}:
        return _references_writer_handle(func.value, writer_aliases) or _is_output_open_call(
            func.value,
            references,
            path_constructors,
        )
    if isinstance(func, ast.Attribute) and func.attr == "dump":
        if isinstance(func.value, ast.Name) and func.value.id == "json":
            if len(node.args) >= 2:
                return _references_writer_handle(node.args[1], writer_aliases) or _is_output_open_call(
                    node.args[1],
                    references,
                    path_constructors,
                )
    if isinstance(func, ast.Name) and func.id == "dump":
        if len(node.args) >= 2:
            return _references_writer_handle(node.args[1], writer_aliases) or _is_output_open_call(
                node.args[1],
                references,
                path_constructors,
            )
    return False


def _is_output_open_call(
    node: ast.AST,
    references: set[str],
    path_constructors: tuple[set[str], set[str]],
) -> bool:
    if not isinstance(node, ast.Call) or not _open_call_writes(node):
        return False
    func = node.func
    if isinstance(func, ast.Name) and func.id == "open":
        return bool(node.args and _is_exact_output_path_reference(node.args[0], references, path_constructors))
    if isinstance(func, ast.Attribute) and func.attr == "open":
        return _is_exact_output_path_reference(func.value, references, path_constructors)
    return False


def _references_writer_handle(node: ast.AST, writer_aliases: set[str]) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id in writer_aliases:
            return True
    return False


def _uses_requests_session(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "Session":
            if isinstance(func.value, ast.Name) and func.value.id == "requests":
                return True
        if isinstance(func, ast.Name) and func.id == "Session":
            return True
    return False


def _uses_beautifulsoup(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "BeautifulSoup":
            return True
        if isinstance(func, ast.Attribute) and func.attr == "BeautifulSoup":
            return True
    return False


def _imports_rendering_browser_tool(tree: ast.AST) -> bool:
    blocked_roots = {"playwright", "selenium"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", 1)[0] in blocked_roots:
                    return True
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.split(".", 1)[0] in blocked_roots:
                return True
    return False


def _env_value_aliases(tree: ast.AST, env_name: str) -> set[str]:
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and _node_reads_env_name(node.value, env_name):
            aliases.update(_assigned_names(node.targets))
        if isinstance(node, ast.AnnAssign) and node.value is not None and _node_reads_env_name(node.value, env_name):
            aliases.update(_assigned_names([node.target]))
    return aliases


def _path_aliases_from_output(
    tree: ast.AST,
    output_aliases: set[str],
    path_constructors: tuple[set[str], set[str]],
) -> set[str]:
    aliases: set[str] = set()
    references = output_aliases | aliases
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            targets: list[ast.expr] = []
            value: ast.AST | None = None
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
                value = node.value
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
                value = node.value
            if value is None:
                continue
            if _is_exact_output_path_reference(value, references, path_constructors):
                before = len(aliases)
                aliases.update(_assigned_names(targets))
                references = output_aliases | aliases
                changed = changed or len(aliases) > before
    return aliases


def _assigned_names(targets: list[ast.expr]) -> set[str]:
    names: set[str] = set()
    for target in targets:
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            names.update(_assigned_names(list(target.elts)))
    return names


def _is_exact_output_path_reference(
    node: ast.AST,
    references: set[str],
    path_constructors: tuple[set[str], set[str]],
) -> bool:
    if isinstance(node, ast.Name):
        return node.id in references
    if _node_reads_env_name(node, "OUTPUT_FILE"):
        return True
    if isinstance(node, ast.Call) and _is_path_constructor_call(node, path_constructors):
        return bool(node.args and _is_exact_output_path_reference(node.args[0], references, path_constructors))
    return False


def _is_path_constructor_call(node: ast.Call, path_constructors: tuple[set[str], set[str]]) -> bool:
    direct_names, module_names = path_constructors
    func = node.func
    if isinstance(func, ast.Name):
        return func.id in direct_names
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "Path"
        and isinstance(func.value, ast.Name)
        and func.value.id in module_names
    )


def _path_constructor_refs(tree: ast.AST) -> tuple[set[str], set[str]]:
    direct_names: set[str] = set()
    module_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "pathlib":
                    module_names.add(alias.asname or "pathlib")
        elif isinstance(node, ast.ImportFrom) and node.module == "pathlib":
            for alias in node.names:
                if alias.name == "Path":
                    direct_names.add(alias.asname or "Path")

    assigned_names = _assigned_name_ids(tree)
    assigned_paths = _assigned_attribute_paths(tree)
    direct_names -= assigned_names
    module_names = {
        name
        for name in module_names
        if name not in assigned_names and f"{name}.Path" not in assigned_paths
    }
    return direct_names, module_names


def _assigned_name_ids(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for target in _assignment_targets(tree):
        names.update(_assigned_names([target]))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


def _assigned_attribute_paths(tree: ast.AST) -> set[str]:
    paths: set[str] = set()
    for target in _assignment_targets(tree):
        path = _attribute_path(target)
        if path:
            paths.add(path)
    return paths


def _assignment_targets(tree: ast.AST) -> list[ast.expr]:
    targets: list[ast.expr] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets.extend(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets.append(node.target)
        elif isinstance(node, ast.AugAssign):
            targets.append(node.target)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            targets.append(node.target)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            targets.extend(item.optional_vars for item in node.items if item.optional_vars is not None)
    return targets


def _open_call_writes(node: ast.Call) -> bool:
    mode = _open_call_mode(node)
    if mode is None:
        return False
    return any(value in mode for value in ("w", "a", "x", "+"))


def _open_call_mode(node: ast.Call) -> str | None:
    if len(node.args) >= 2:
        return _literal_string(node.args[1])
    for keyword in node.keywords:
        if keyword.arg == "mode":
            return _literal_string(keyword.value)
    return None


# HTML analysis helpers


def _common_containers(soup: BeautifulSoup) -> list[dict[str, str]]:
    containers: list[dict[str, str]] = []
    for tag_name in ("main", "article", "section", "div", "li"):
        for element in soup.find_all(tag_name, class_=True)[:8]:
            classes = " ".join(element.get("class", []))
            preview = _collapse_text(element.get_text(" "))[:120]
            if classes and preview:
                containers.append({"tag": tag_name, "class": classes, "text_preview": preview})
    return containers


def _sample_links(soup: BeautifulSoup, url: str) -> list[dict[str, str]]:
    links: list[dict[str, str]] = []
    for link in soup.find_all("a", href=True)[:30]:
        href = str(link["href"]).strip()
        if url:
            href = urljoin(url, href)
        label = _collapse_text(link.get_text(" "))[:80]
        if href:
            links.append({"href": href, "text": label})
    return links


def _json_ld_types(soup: BeautifulSoup) -> list[Any]:
    types: list[Any] = []
    for script in soup.find_all("script", attrs={"type": re.compile("ld\\+json", re.I)}):
        raw = script.string or script.get_text()
        for item in _json_ld_items(raw or ""):
            item_type = item.get("@type") if isinstance(item, dict) else None
            if item_type:
                types.append(item_type)
    return types


def _visible_text(soup: BeautifulSoup) -> str:
    copied = BeautifulSoup(str(soup), "lxml")
    for tag in copied(["script", "style", "noscript", "svg"]):
        tag.decompose()
    return _collapse_text(copied.get_text(" "))


def _collapse_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _json_ld_items(raw: str) -> list[dict[str, Any]]:
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return _flatten_json_ld(loaded)


def _flatten_json_ld(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        items: list[dict[str, Any]] = []
        for entry in value:
            items.extend(_flatten_json_ld(entry))
        return items
    if isinstance(value, dict):
        items = [value]
        graph = value.get("@graph")
        if isinstance(graph, list):
            items.extend(_flatten_json_ld(graph))
        return items
    return []


def _likely_job_sections(soup: BeautifulSoup) -> list[dict[str, str]]:
    sections: list[dict[str, str]] = []
    for element in soup.find_all(["main", "article", "section", "div", "ul"]):
        attributes = _section_hint_attributes(element)
        text = _collapse_text(element.get_text(" "))
        if SECTION_HINT_PATTERN.search(attributes):
            sections.append({
                "tag": element.name or "",
                "class": " ".join(element.get("class", []))[:160],
                "text_preview": text[:180],
            })
        if len(sections) >= 12:
            break
    return sections


def _section_hint_attributes(element: Any) -> str:
    values: list[str] = []
    classes = element.get("class", [])
    if isinstance(classes, list):
        values.extend(str(value) for value in classes)
    elif classes:
        values.append(str(classes))

    for key, value in element.attrs.items():
        if key == "class":
            continue
        if key == "id" or key == "aria-label" or key.startswith("data-"):
            if isinstance(value, list):
                values.extend(str(item) for item in value)
            elif value:
                values.append(str(value))

    return " ".join(values)


def _detect_anti_scraping(html: str, text: str) -> dict[str, Any]:
    lowered = html.lower()
    mechanisms: list[str] = []
    if _has_cloudflare_challenge(lowered):
        mechanisms.append("Cloudflare")
    if any(value in lowered for value in ("hcaptcha", "g-recaptcha", "recaptcha")):
        mechanisms.append("CAPTCHA widget")
    script_count = len(BeautifulSoup(html or "", "lxml").find_all("script"))
    if script_count >= 8 and len(text) < 800:
        mechanisms.append("JavaScript Rendering")
    return {
        "has_anti_scraping": bool(mechanisms),
        "detected_mechanisms": mechanisms,
    }


def _has_cloudflare_challenge(lowered_html: str) -> bool:
    if CLOUDFLARE_CHALLENGE_PATTERN.search(lowered_html):
        return True
    return "cloudflare" in lowered_html and (
        "checking your browser" in lowered_html
        or "attention required!" in lowered_html
        or "just a moment..." in lowered_html
    )


def _recommended_strategy(anti_scraping: dict[str, Any], json_ld_types: list[Any], text: str) -> str:
    mechanisms = set(anti_scraping.get("detected_mechanisms", []))
    if "Cloudflare" in mechanisms:
        return "Do not bypass protection. Use another official/ATS URL if available."
    if "CAPTCHA widget" in mechanisms and len(text) < 700:
        return "Do not solve CAPTCHA. Collect only accessible evidence and leave semantic judgment to job_verifier."
    if any("JobPosting" in str(item) for item in json_ld_types):
        return "Collect JSON-LD JobPosting, meta fields, visible text, links, and headings into the context schema."
    if "JavaScript Rendering" in mechanisms:
        return "Use browser_extract_job_page for rendered fallback. Keep generated crawler code limited to requests.Session plus BeautifulSoup/lxml for static accessible evidence."
    if len(text) > 1000:
        return "Use requests.Session plus BeautifulSoup to collect protocol fields, visible text, links, and headings."
    return "Page is sparse; collect available context and leave final semantic verification to job_verifier."


def _strip_code_fence(code: str) -> str:
    text = code.strip()
    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip() + "\n"
    return text + "\n"


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("._")
    return safe[:80] or "job"


CRAWLER_CODE_TOOLS = [
    analyze_job_html_structure,
    save_job_crawler_code,
    validate_job_crawler_code,
    build_job_crawler_run_command,
]
