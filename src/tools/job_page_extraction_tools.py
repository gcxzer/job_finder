from __future__ import annotations

import hashlib
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import tool

from src.configs import CONFIG
from src.tools.url_safety import public_http_url_error

WORKSPACE_DIR = CONFIG.workspace.root_dir
PAGE_CACHE_DIR = CONFIG.workspace.page_cache_dir
CONTAINER_WORKSPACE = CONFIG.docker.container_workspace_dir
SCHEMA_VERSION = "job_extraction_context_v1"
MAX_HTML_PREVIEW_CHARS = 1200
MAX_TEXT_PREVIEW_CHARS = 2000
MAX_VISIBLE_TEXT_CHARS = 6000
MAX_CANDIDATE_LINKS = 80
MAX_HEADINGS = 40
SPARSE_CAPTCHA_TEXT_CHARS = 80
MAX_HTML_BYTES = 2_000_000

DESKTOP_USER_AGENTS = [
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) "
        "Gecko/20100101 Firefox/125.0"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) "
        "Version/17.4 Safari/605.1.15"
    ),
]

TECHNICAL_BLOCKED_PATTERNS = {
    "cloudflare_challenge": r"cf-chl|__cf_chl_|/cdn-cgi/challenge-platform|cf-browser-verification",
    "captcha_widget": r"hcaptcha|g-recaptcha|recaptcha",
}


@tool
def fetch_job_page(url: str) -> dict[str, Any]:
    """Fetch a job page with HTTP and cache the HTML for later extraction."""
    clean_url = url.strip()
    url_error = public_http_url_error(clean_url)
    if url_error:
        return {"success": False, "url": url, "error": f"Unsafe URL: {url_error}"}

    try:
        response, attempt_count = _fetch_with_reference_crawler(clean_url)
        html = _response_text(response)
        html_file = _write_cached_html(clean_url, html, suffix="crawler")
        text = _visible_text(html)
        blocked = _detect_blocked(html, response.status_code)
        is_closed = response.status_code in {404, 410}
        is_js_heavy = _is_js_heavy(html, text)
        verification_status = _fetch_verification_status(blocked["reason"], is_closed)

        return {
            "success": response.status_code < 400 and not blocked["is_blocked"] and not is_closed,
            "url": clean_url,
            "final_url": str(response.url),
            "status_code": response.status_code,
            "html_file": str(html_file),
            "html_preview": html[:MAX_HTML_PREVIEW_CHARS],
            "text_preview": text[:MAX_TEXT_PREVIEW_CHARS],
            "text_length": len(text),
            "is_js_heavy": is_js_heavy,
            "anti_scraping": _detect_anti_scraping(html),
            "verification_status": verification_status,
            "blocked_reason": blocked["reason"],
            "attempt_count": attempt_count,
            "error": None if response.status_code < 400 else f"HTTP {response.status_code}",
        }
    except Exception as error:
        return {
            "success": False,
            "url": clean_url,
            "final_url": clean_url,
            "status_code": 0,
            "html_file": "",
            "html_preview": "",
            "text_preview": "",
            "text_length": 0,
            "is_js_heavy": False,
            "anti_scraping": {
                "detected_mechanisms": [],
                "has_anti_scraping": False,
                "recommendations": [],
            },
            "verification_status": "unverified",
            "blocked_reason": "",
            "attempt_count": 0,
            "error": str(error),
        }


@tool
def extract_job_posting(url: str, html: str = "", html_file: str = "") -> dict[str, Any]:
    """Build a standard extraction context from HTML or a cached HTML file."""
    content = _load_html(html=html, html_file=html_file)
    if not content:
        return _unavailable_context(
            url=url,
            extraction_method="crawler",
            html_file=html_file,
            error="No HTML content provided.",
        )

    return _extract_job_posting(
        url=url,
        html=content,
        extraction_method="crawler",
        html_file=html_file,
    )


@tool
def browser_extract_job_page(url: str) -> dict[str, Any]:
    """Use Playwright as a fallback to build a rendered job page context."""
    clean_url = url.strip()
    url_error = public_http_url_error(clean_url)
    if url_error:
        return _unavailable_context(
            url=clean_url,
            extraction_method="unavailable",
            error=f"Unsafe URL: {url_error}",
        )

    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except ImportError as error:
        return _unavailable_context(
            url=clean_url,
            extraction_method="unavailable",
            error=f"Playwright is not installed: {error}",
        )

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(
                user_agent=_random_user_agent(),
                extra_http_headers={
                    "Accept-Language": "en-US,en;q=0.9,de;q=0.8",
                    "Upgrade-Insecure-Requests": "1",
                },
            )
            route_guard = _PlaywrightUrlGuard()
            page.route("**/*", route_guard.handle)
            try:
                response = page.goto(clean_url, wait_until="domcontentloaded", timeout=25000)
                page.wait_for_timeout(2500)
                final_url = page.url
                final_url_error = public_http_url_error(final_url)
                if final_url_error:
                    browser.close()
                    return _unavailable_context(
                        url=clean_url,
                        extraction_method="browser",
                        error=f"Unsafe final URL: {final_url_error}",
                    )
                status_code = response.status if response is not None else 0
                html = _limit_text_bytes(page.content(), MAX_HTML_BYTES)
            except PlaywrightTimeoutError as error:
                browser.close()
                return _unavailable_context(
                    url=clean_url,
                    extraction_method="browser",
                    error=f"Browser timeout: {error}",
                )
            finally:
                try:
                    browser.close()
                except Exception:
                    pass

        html_file = _write_cached_html(clean_url, html, suffix="browser")
        return _extract_job_posting(
            url=final_url,
            html=html,
            extraction_method="browser",
            html_file=str(html_file),
            status_code=status_code,
        )
    except Exception as error:
        return _unavailable_context(
            url=clean_url,
            extraction_method="browser",
            error=str(error),
        )


def _fetch_with_reference_crawler(url: str) -> tuple[httpx.Response, int]:
    """Fetch using the same crawler posture as references/tools.py."""
    last_error: Exception | None = None
    current_url = url
    for attempt in range(1, 3):
        try:
            response = _fetch_with_checked_redirects(current_url)
            if response.status_code not in {403, 429} or attempt == 2:
                return response, attempt
        except Exception as error:
            last_error = error
            if attempt == 2:
                raise
    if last_error is not None:
        raise last_error
    raise RuntimeError("Fetch failed without a response.")


def _fetch_with_checked_redirects(url: str, *, max_redirects: int = 8) -> httpx.Response:
    current_url = url
    for _redirect_index in range(max_redirects + 1):
        url_error = public_http_url_error(current_url)
        if url_error:
            raise ValueError(f"Unsafe URL: {url_error}")

        headers = _safe_headers(current_url)
        with httpx.Client(
            headers=headers,
            follow_redirects=False,
            timeout=httpx.Timeout(25.0, connect=10.0),
        ) as client:
            with client.stream("GET", current_url) as response:
                body = _read_limited_response_body(response, max_bytes=MAX_HTML_BYTES)
                checked_response = httpx.Response(
                    status_code=response.status_code,
                    headers=response.headers,
                    content=body,
                    request=response.request,
                    extensions=response.extensions,
                )

        if checked_response.status_code not in {301, 302, 303, 307, 308}:
            return checked_response

        location = checked_response.headers.get("location", "").strip()
        if not location:
            return checked_response
        next_url = urljoin(str(checked_response.url), location)
        next_url_error = public_http_url_error(next_url)
        if next_url_error:
            raise ValueError(f"Unsafe redirect blocked: {next_url_error}")
        current_url = next_url

    raise ValueError(f"Too many redirects while fetching {url}.")


class _PlaywrightUrlGuard:
    def __init__(self) -> None:
        self._host_errors: dict[str, str] = {}

    def handle(self, route: Any, request: Any) -> None:
        url = str(getattr(request, "url", "") or "")
        error = self._cached_error(url)
        if error:
            route.abort()
            return
        route.continue_()

    def _cached_error(self, url: str) -> str:
        host = _url_host_key(url)
        if host not in self._host_errors:
            self._host_errors[host] = public_http_url_error(url)
        return self._host_errors[host]


def _url_host_key(url: str) -> str:
    try:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        return f"{parsed.scheme.lower()}://{str(parsed.hostname or '').lower()}"
    except Exception:
        return url


def _safe_headers(url: str) -> dict[str, str]:
    """Return desktop-browser headers inspired by references.tools.get_safe_headers."""
    return {
        "User-Agent": _random_user_agent(),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,image/apng,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9,de;q=0.8",
        # Keep br out, matching the reference crawler, so decoding stays predictable.
        "Accept-Encoding": "gzip, deflate",
        "Referer": url,
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }


def _random_user_agent() -> str:
    return random.choice(DESKTOP_USER_AGENTS)


def _response_text(response: httpx.Response) -> str:
    try:
        return response.text or ""
    except UnicodeDecodeError:
        return response.content.decode("gbk", errors="ignore")


def _read_limited_response_body(response: httpx.Response, *, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_bytes():
        total += len(chunk)
        if total > max_bytes:
            raise ValueError(f"Response body is too large; limit is {max_bytes} bytes.")
        chunks.append(chunk)
    return b"".join(chunks)


def _limit_text_bytes(text: str, max_bytes: int) -> str:
    encoded = text.encode("utf-8", errors="ignore")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def _extract_job_posting(
    url: str,
    html: str,
    extraction_method: str,
    html_file: str = "",
    status_code: int = 200,
) -> dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")
    text = _visible_text(html)
    blocked = _detect_blocked(html, status_code=status_code)
    technical_status = _technical_status(status_code=status_code, blocked_reason=blocked["reason"])
    standard_extraction = _standard_extraction(soup, url)
    technical_signals = _technical_signals(html=html, text=text, status_code=status_code, blocked=blocked)

    return {
        "success": technical_status == "readable",
        "schema_version": SCHEMA_VERSION,
        "url": url,
        "final_url": url,
        "extraction_method": extraction_method,
        "technical_status": technical_status,
        "verification_status": _context_verification_status(technical_status),
        "standard_extraction": standard_extraction,
        "page_context": {
            "visible_text": text[:MAX_VISIBLE_TEXT_CHARS],
            "visible_text_truncated": len(text) > MAX_VISIBLE_TEXT_CHARS,
            "text_length": len(text),
            "candidate_links": _candidate_links(soup, url),
            "headings": _headings(soup),
            "html_file": html_file,
        },
        "technical_signals": technical_signals,
        "verified_at": _now_iso(),
        "error": None if technical_status == "readable" else technical_status,
    }


def _unavailable_context(
    *,
    url: str,
    extraction_method: str,
    error: str,
    html_file: str = "",
) -> dict[str, Any]:
    return {
        "success": False,
        "schema_version": SCHEMA_VERSION,
        "url": url,
        "final_url": url,
        "extraction_method": extraction_method,
        "technical_status": "unavailable",
        "verification_status": "unverified",
        "standard_extraction": {
            "canonical_url": url,
            "html_title": "",
            "meta": {},
            "json_ld_jobposting": {},
        },
        "page_context": {
            "visible_text": "",
            "visible_text_truncated": False,
            "text_length": 0,
            "candidate_links": [],
            "headings": [],
            "html_file": html_file,
        },
        "technical_signals": {
            "status_code": 0,
            "detected_mechanisms": [],
            "has_anti_scraping": False,
            "is_js_heavy": False,
            "is_blocked": False,
            "blocked_reason": "",
        },
        "verified_at": _now_iso(),
        "error": error,
    }


def _standard_extraction(soup: BeautifulSoup, url: str) -> dict[str, Any]:
    jobposting = _find_jobposting_json_ld(soup)
    return {
        "canonical_url": _canonical_url(soup, url),
        "html_title": _collapse_text(soup.title.string if soup.title else ""),
        "meta": {
            "og:title": _meta(soup, "og:title"),
            "twitter:title": _meta(soup, "twitter:title"),
            "og:description": _meta(soup, "og:description"),
            "description": _meta(soup, "description"),
            "og:site_name": _meta(soup, "og:site_name"),
            "og:url": _meta(soup, "og:url"),
        },
        "json_ld_jobposting": _extract_from_json_ld(jobposting, url) if jobposting else {},
    }


def _extract_from_json_ld(jobposting: dict[str, Any], page_url: str) -> dict[str, Any]:
    organization = jobposting.get("hiringOrganization") or {}
    if isinstance(organization, list):
        organization = organization[0] if organization else {}
    apply_url = _safe_public_url_or_empty(
        _as_text(jobposting.get("url") or jobposting.get("sameAs")),
        page_url,
    )

    return {
        "title": _as_text(jobposting.get("title")),
        "company": _as_text(organization.get("name") if isinstance(organization, dict) else organization),
        "location": _location_text(jobposting.get("jobLocation")),
        "salary": _salary_text(jobposting.get("baseSalary")),
        "description": _html_to_text(_as_text(jobposting.get("description"))),
        "requirements": _json_ld_requirements(jobposting),
        "apply_url": apply_url,
        "posted_date": _as_text(jobposting.get("datePosted")),
        "valid_through": _as_text(jobposting.get("validThrough")),
        "employment_type": _as_text(jobposting.get("employmentType")),
    }


def _candidate_links(soup: BeautifulSoup, url: str) -> list[dict[str, str]]:
    links: list[dict[str, str]] = []
    seen: set[str] = set()
    for link in soup.find_all("a", href=True):
        raw_href = str(link["href"]).strip()
        if not raw_href or raw_href.startswith(("#", "javascript:")):
            continue
        href = urljoin(url, raw_href)
        if _unsafe_output_url(href):
            continue
        if href in seen:
            continue
        seen.add(href)
        links.append(
            {
                "href": href,
                "text": _collapse_text(link.get_text(" "))[:160],
                "aria_label": _collapse_text(link.get("aria-label") or "")[:160],
                "title": _collapse_text(link.get("title") or "")[:160],
            }
        )
        if len(links) >= MAX_CANDIDATE_LINKS:
            break
    return links


def _headings(soup: BeautifulSoup) -> list[dict[str, str]]:
    headings: list[dict[str, str]] = []
    for heading in soup.find_all(["h1", "h2", "h3", "h4"]):
        text = _collapse_text(heading.get_text(" "))
        if not text:
            continue
        headings.append({"tag": heading.name or "", "text": text[:180]})
        if len(headings) >= MAX_HEADINGS:
            break
    return headings


def _write_cached_html(url: str, html: str, suffix: str) -> Path:
    PAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    path = PAGE_CACHE_DIR / f"{key}_{suffix}.html"
    path.write_text(_limit_text_bytes(html, MAX_HTML_BYTES), encoding="utf-8", errors="ignore")
    return path


def _load_html(*, html: str, html_file: str) -> str:
    if html:
        return html
    if html_file:
        path = _resolve_workspace_file(html_file)
        if path is not None and path.exists() and path.is_file():
            if path.stat().st_size > MAX_HTML_BYTES:
                return ""
            return path.read_text(encoding="utf-8", errors="ignore")
    return ""


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


def _visible_text(html: str) -> str:
    soup = BeautifulSoup(html or "", "lxml")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    return _collapse_text(soup.get_text(" "))


def _html_to_text(value: str) -> str:
    return _visible_text(value)


def _collapse_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _detect_blocked(html: str, status_code: int) -> dict[str, Any]:
    lowered = (html or "").lower()
    visible_text_length = len(_visible_text(html).strip())
    if status_code in {401, 403, 429, 999}:
        return {"is_blocked": True, "reason": f"HTTP {status_code}"}
    if _has_cloudflare_challenge(lowered):
        return {"is_blocked": True, "reason": "cloudflare"}
    if _has_captcha_widget(lowered) and visible_text_length < SPARSE_CAPTCHA_TEXT_CHARS:
        return {"is_blocked": True, "reason": "captcha"}
    return {"is_blocked": False, "reason": ""}


def _detect_anti_scraping(html: str) -> dict[str, Any]:
    text = _visible_text(html)
    signals = _technical_signals(
        html=html,
        text=text,
        status_code=200,
        blocked=_detect_blocked(html, status_code=200),
    )
    recommendations: list[str] = []

    if signals["has_cloudflare"]:
        recommendations.append("Do not bypass Cloudflare; mark as access_blocked and keep for manual review.")
    if signals["has_captcha_widget"] and signals["is_sparse_captcha_page"]:
        recommendations.append("Do not bypass CAPTCHA; mark as access_blocked and keep for manual review.")
    if signals["is_js_heavy"]:
        recommendations.append("Use browser_extract_job_page as a fallback.")
    if not recommendations:
        recommendations = [
            "Use desktop browser headers.",
            "Keep full HTML in cache and pass html_file to extraction tools.",
            "Use browser fallback only when crawler text is too short or extraction context is sparse.",
        ]

    return {
        "detected_mechanisms": signals["detected_mechanisms"],
        "has_anti_scraping": signals["has_anti_scraping"],
        "recommendations": recommendations,
    }


def _technical_signals(
    *,
    html: str,
    text: str,
    status_code: int,
    blocked: dict[str, Any],
) -> dict[str, Any]:
    lowered = (html or "").lower()
    has_cloudflare = _has_cloudflare_challenge(lowered)
    has_captcha_widget = _has_captcha_widget(lowered)
    is_sparse_captcha_page = has_captcha_widget and len(text.strip()) < SPARSE_CAPTCHA_TEXT_CHARS
    is_js_heavy = _is_js_heavy(html, text)
    detected = []
    if status_code in {401, 403, 429, 999}:
        detected.append(f"HTTP {status_code}")
    if status_code in {404, 410}:
        detected.append(f"HTTP {status_code} closed")
    if has_cloudflare:
        detected.append("Cloudflare")
    if has_captcha_widget:
        detected.append("CAPTCHA widget")
    if is_js_heavy:
        detected.append("JavaScript Rendering")

    return {
        "status_code": status_code,
        "detected_mechanisms": detected,
        "has_anti_scraping": bool(
            status_code in {401, 403, 429, 999}
            or has_cloudflare
            or has_captcha_widget
            or is_js_heavy
        ),
        "has_cloudflare": has_cloudflare,
        "has_captcha_widget": has_captcha_widget,
        "is_sparse_captcha_page": is_sparse_captcha_page,
        "is_js_heavy": is_js_heavy,
        "is_blocked": bool(blocked.get("is_blocked")),
        "blocked_reason": str(blocked.get("reason") or ""),
    }


def _has_captcha_widget(lowered_html: str) -> bool:
    return bool(re.search(TECHNICAL_BLOCKED_PATTERNS["captcha_widget"], lowered_html, flags=re.IGNORECASE))


def _has_cloudflare_challenge(lowered_html: str) -> bool:
    if re.search(TECHNICAL_BLOCKED_PATTERNS["cloudflare_challenge"], lowered_html, flags=re.IGNORECASE):
        return True
    return "cloudflare" in lowered_html and (
        "checking your browser" in lowered_html
        or "attention required!" in lowered_html
        or "just a moment..." in lowered_html
    )


def _technical_status(status_code: int, blocked_reason: str) -> str:
    if status_code in {404, 410}:
        return "closed_http"
    if blocked_reason == "HTTP 401":
        return "login_required"
    if blocked_reason:
        return "access_blocked"
    return "readable"


def _context_verification_status(technical_status: str) -> str:
    if technical_status == "closed_http":
        return "closed"
    if technical_status == "login_required":
        return "login_required"
    if technical_status == "access_blocked":
        return "access_blocked"
    return "unverified"


def _fetch_verification_status(blocked_reason: str, is_closed: bool) -> str:
    if is_closed:
        return "closed"
    if blocked_reason:
        return _blocked_verification_status(blocked_reason)
    return "unverified"


def _blocked_verification_status(reason: str) -> str:
    if reason == "HTTP 401":
        return "login_required"
    return "access_blocked"


def _is_js_heavy(html: str, text: str) -> bool:
    soup = BeautifulSoup(html or "", "lxml")
    script_count = len(soup.find_all("script"))
    return script_count >= 8 and len(text.strip()) < 800


def _find_jobposting_json_ld(soup: BeautifulSoup) -> dict[str, Any] | None:
    for script in soup.find_all("script", attrs={"type": re.compile("ld\\+json", re.I)}):
        raw = script.string or script.get_text()
        if not raw:
            continue
        for item in _json_ld_items(raw):
            item_type = item.get("@type") if isinstance(item, dict) else None
            types = item_type if isinstance(item_type, list) else [item_type]
            if any(str(value).lower() == "jobposting" for value in types):
                return item
    return None


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


def _meta(soup: BeautifulSoup, name: str) -> str:
    selectors = [
        {"property": name},
        {"name": name},
    ]
    for selector in selectors:
        tag = soup.find("meta", attrs=selector)
        if tag and tag.get("content"):
            return str(tag["content"]).strip()
    return ""


def _canonical_url(soup: BeautifulSoup, url: str) -> str:
    canonical = soup.find("link", rel=lambda value: value and "canonical" in value)
    if canonical and canonical.get("href"):
        href = urljoin(url, str(canonical["href"]).strip())
        return href if not _unsafe_output_url(href) else url
    og_url = _meta(soup, "og:url")
    if og_url:
        href = urljoin(url, og_url)
        return href if not _unsafe_output_url(href) else url
    return url


def _safe_public_url_or_empty(value: str, page_url: str) -> str:
    clean_value = str(value or "").strip()
    if not clean_value:
        return ""
    href = urljoin(page_url, clean_value)
    return "" if _unsafe_output_url(href) else href


def _unsafe_output_url(url: str) -> bool:
    return bool(public_http_url_error(url, resolve_dns=False))


def _location_text(value: Any) -> str:
    if isinstance(value, list):
        return "; ".join(filter(None, [_location_text(item) for item in value]))
    if isinstance(value, dict):
        address = value.get("address")
        if isinstance(address, dict):
            parts = [
                address.get("streetAddress"),
                address.get("addressLocality"),
                address.get("addressRegion"),
                address.get("postalCode"),
                address.get("addressCountry"),
            ]
            return ", ".join(str(part) for part in parts if part)
        return _as_text(value.get("name") or address)
    return _as_text(value)


def _salary_text(value: Any) -> str:
    if isinstance(value, dict):
        currency = value.get("currency") or ""
        amount = value.get("value") or value.get("minValue") or ""
        if isinstance(amount, dict):
            min_value = amount.get("minValue")
            max_value = amount.get("maxValue")
            unit = amount.get("unitText") or ""
            if min_value and max_value:
                return f"{currency} {min_value}-{max_value} {unit}".strip()
            if amount.get("value"):
                return f"{currency} {amount['value']} {unit}".strip()
        if amount:
            return f"{currency} {amount}".strip()
    return _as_text(value)


def _json_ld_requirements(jobposting: dict[str, Any]) -> list[str]:
    fields = [
        "qualifications",
        "skills",
        "experienceRequirements",
        "responsibilities",
        "educationRequirements",
    ]
    values = [_as_text(jobposting.get(field)) for field in fields if jobposting.get(field)]
    return [_collapse_text(_html_to_text(value)) for value in values if value]


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(_as_text(item) for item in value if item)
    if isinstance(value, dict):
        return _as_text(value.get("name") or value.get("value") or value.get("@id") or "")
    return str(value).strip()


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


JOB_PAGE_TOOLS = [
    fetch_job_page,
    extract_job_posting,
    browser_extract_job_page,
]
