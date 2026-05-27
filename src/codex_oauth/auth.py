from __future__ import annotations

import base64
import json
import os
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from codex_oauth.types import CodexCredentials


DEFAULT_CODEX_AUTH_PATH = Path.cwd() / ".codex_oauth" / "auth" / "codex.json"
DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
CODEX_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_OAUTH_ISSUER = "https://auth.openai.com"
CODEX_OAUTH_TOKEN_URL = f"{CODEX_OAUTH_ISSUER}/oauth/token"
CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS = 120
CODEX_CLIENT_HEADERS = {
    "User-Agent": "codex_cli_rs/0.0.0 (codex-oauth)",
    "originator": "codex_cli_rs",
}
_AUTH_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class CodexAuthStatus:
    logged_in: bool = False
    auth_mode: str = ""
    plan_type: str = ""
    account_id: str = ""
    account_email: str = ""
    last_refresh: str = ""
    auth_store_path: str = ""

    def to_public_dict(self) -> dict[str, object]:
        return {
            "provider": "codex-oauth",
            "loggedIn": self.logged_in,
            "authMode": self.auth_mode,
            "planType": self.plan_type,
            "accountId": self.account_id,
            "accountEmail": self.account_email,
            "lastRefresh": self.last_refresh,
            "authStorePath": self.auth_store_path,
        }


@dataclass(frozen=True, slots=True)
class CodexDeviceAuthStart:
    user_code: str
    device_auth_id: str
    verification_uri: str
    interval: int = 5

    def to_public_dict(self) -> dict[str, object]:
        return {
            "status": "started",
            "userCode": self.user_code,
            "deviceAuthId": self.device_auth_id,
            "verificationUri": self.verification_uri,
            "interval": self.interval,
        }


@dataclass(frozen=True, slots=True)
class CodexDeviceAuthPoll:
    status: str
    credentials: CodexCredentials | None = None

    def to_public_dict(self) -> dict[str, object]:
        return {"status": self.status}


class CodexAuthError(RuntimeError):
    def __init__(self, message: str, *, code: str = "codex_auth_error") -> None:
        super().__init__(message)
        self.code = code


class CodexAuthStore:
    """Local Codex OAuth token store."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path).expanduser() if path is not None else default_codex_auth_path()

    def read_credentials(self) -> CodexCredentials:
        if not self.path.exists():
            return CodexCredentials()
        try:
            with _AUTH_LOCK:
                data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return CodexCredentials()
        return CodexCredentials.from_dict(data if isinstance(data, dict) else {})

    def write_credentials(self, credentials: CodexCredentials) -> None:
        payload = credentials.to_dict()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _secure_dir(self.path.parent)
        with _AUTH_LOCK:
            fd, tmp_path = tempfile.mkstemp(dir=str(self.path.parent), prefix=".codex_", suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as file:
                    json.dump(payload, file, ensure_ascii=False, indent=2)
                    file.write("\n")
                    file.flush()
                    os.fsync(file.fileno())
                os.replace(tmp_path, self.path)
                _secure_file(self.path)
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

    def clear(self) -> None:
        with _AUTH_LOCK:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass

    def status(self) -> CodexAuthStatus:
        credentials = self.read_credentials()
        account_id = credentials.account_id or _chatgpt_account_id(credentials.access_token)
        return CodexAuthStatus(
            logged_in=credentials.configured,
            auth_mode="chatgpt" if credentials.configured else "",
            plan_type=credentials.plan_type,
            account_id=account_id,
            account_email=credentials.account_email,
            last_refresh=credentials.last_refresh,
            auth_store_path=str(self.path),
        )

    def runtime_credentials(self, *, refresh_if_expiring: bool = True) -> CodexCredentials:
        credentials = self.read_credentials()
        if not credentials.configured:
            return credentials
        if refresh_if_expiring and _access_token_is_expiring(credentials.access_token):
            credentials = refresh_codex_credentials(credentials)
            self.write_credentials(credentials)
        return credentials


class CodexDeviceAuthClient:
    def start(self) -> CodexDeviceAuthStart:
        status, payload = _post_json(
            f"{CODEX_OAUTH_ISSUER}/api/accounts/deviceauth/usercode",
            {"client_id": CODEX_OAUTH_CLIENT_ID},
        )
        if status != 200:
            raise CodexAuthError(f"Device code request returned status {status}.", code="device_code_request_error")

        user_code = _string_value(payload.get("user_code"))
        device_auth_id = _string_value(payload.get("device_auth_id"))
        interval = max(3, _int_value(payload.get("interval"), 5))
        if not user_code or not device_auth_id:
            raise CodexAuthError("Device code response was missing required fields.", code="device_code_incomplete")
        return CodexDeviceAuthStart(
            user_code=user_code,
            device_auth_id=device_auth_id,
            verification_uri=f"{CODEX_OAUTH_ISSUER}/codex/device",
            interval=interval,
        )

    def poll(self, *, device_auth_id: str, user_code: str) -> CodexDeviceAuthPoll:
        status, payload = _post_json(
            f"{CODEX_OAUTH_ISSUER}/api/accounts/deviceauth/token",
            {"device_auth_id": device_auth_id, "user_code": user_code},
        )
        if status in {403, 404}:
            return CodexDeviceAuthPoll(status="pending")
        if status != 200:
            raise CodexAuthError(f"Device auth polling returned status {status}.", code="device_code_poll_error")

        authorization_code = _string_value(payload.get("authorization_code"))
        code_verifier = _string_value(payload.get("code_verifier"))
        if not authorization_code or not code_verifier:
            raise CodexAuthError(
                "Device auth response was missing authorization_code or code_verifier.",
                code="device_code_incomplete_exchange",
            )
        return CodexDeviceAuthPoll(
            status="connected",
            credentials=exchange_codex_device_code(authorization_code, code_verifier),
        )


def exchange_codex_device_code(authorization_code: str, code_verifier: str) -> CodexCredentials:
    status, payload = _post_form(
        CODEX_OAUTH_TOKEN_URL,
        {
            "grant_type": "authorization_code",
            "code": authorization_code,
            "redirect_uri": f"{CODEX_OAUTH_ISSUER}/deviceauth/callback",
            "client_id": CODEX_OAUTH_CLIENT_ID,
            "code_verifier": code_verifier,
        },
    )
    if status != 200:
        raise CodexAuthError(f"Token exchange returned status {status}.", code="token_exchange_error")
    access_token = _string_value(payload.get("access_token"))
    if not access_token:
        raise CodexAuthError("Token exchange did not return an access_token.", code="token_exchange_no_access_token")
    return _credentials_from_token_payload(payload, source="device-code")


def refresh_codex_credentials(credentials: CodexCredentials) -> CodexCredentials:
    if not credentials.refresh_token:
        return credentials
    status, payload = _post_form(
        CODEX_OAUTH_TOKEN_URL,
        {
            "grant_type": "refresh_token",
            "refresh_token": credentials.refresh_token,
            "client_id": CODEX_OAUTH_CLIENT_ID,
        },
    )
    if status != 200:
        raise CodexAuthError(f"Codex token refresh returned status {status}.", code="codex_refresh_failed")
    access_token = _string_value(payload.get("access_token"))
    if not access_token:
        raise CodexAuthError("Codex token refresh response was missing access_token.", code="codex_refresh_missing_access_token")
    refresh_token = _string_value(payload.get("refresh_token")) or credentials.refresh_token
    return CodexCredentials(
        access_token=access_token,
        refresh_token=refresh_token,
        id_token=_string_value(payload.get("id_token")) or credentials.id_token,
        account_id=credentials.account_id or _chatgpt_account_id(access_token),
        account_email=credentials.account_email,
        plan_type=credentials.plan_type,
        expires_at=_expires_at(access_token),
        last_refresh=_now_iso(),
        source=credentials.source or "device-code",
        base_url=credentials.base_url or DEFAULT_CODEX_BASE_URL,
    )


def codex_default_headers(access_token: str) -> dict[str, str]:
    headers = dict(CODEX_CLIENT_HEADERS)
    account_id = _chatgpt_account_id(access_token)
    if account_id:
        headers["ChatGPT-Account-ID"] = account_id
    return headers


def default_codex_auth_path() -> Path:
    override = os.environ.get("CODEX_OAUTH_AUTH_PATH", "").strip()
    return Path(override).expanduser() if override else DEFAULT_CODEX_AUTH_PATH


def _secure_dir(path: Path) -> None:
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass


def _secure_file(path: Path) -> None:
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _credentials_from_token_payload(payload: dict[str, Any], *, source: str) -> CodexCredentials:
    access_token = _string_value(payload.get("access_token"))
    return CodexCredentials(
        access_token=access_token,
        refresh_token=_string_value(payload.get("refresh_token")),
        id_token=_string_value(payload.get("id_token")),
        account_id=_chatgpt_account_id(access_token),
        expires_at=_expires_at(access_token),
        last_refresh=_now_iso(),
        source=source,
        base_url=DEFAULT_CODEX_BASE_URL,
    )


def _post_json(url: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    data = json.dumps(payload).encode("utf-8")
    return _request_json(url, data=data, headers={"Content-Type": "application/json", "Accept": "application/json"})


def _post_form(url: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    data = urlencode(payload).encode("utf-8")
    return _request_json(url, data=data, headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"})


def _request_json(url: str, *, data: bytes, headers: dict[str, str]) -> tuple[int, dict[str, Any]]:
    request = Request(url, data=data, headers={**CODEX_CLIENT_HEADERS, **headers}, method="POST")
    try:
        with urlopen(request, timeout=15) as response:
            return response.status, _parse_json(response.read())
    except HTTPError as error:
        return error.code, _parse_json(error.read())
    except OSError as error:
        raise CodexAuthError(f"Codex auth request failed: {error}", code="codex_auth_request_failed") from error


def _parse_json(data: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(data.decode("utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _access_token_is_expiring(token: str, skew_seconds: int = CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS) -> bool:
    claims = _jwt_claims(token)
    exp = _int_value(claims.get("exp"), 0)
    if not exp:
        return False
    return datetime.now(timezone.utc).timestamp() >= exp - skew_seconds


def _chatgpt_account_id(token: str) -> str:
    claims = _jwt_claims(token)
    auth = claims.get("https://api.openai.com/auth")
    if isinstance(auth, dict):
        return _string_value(auth.get("chatgpt_account_id"))
    return ""


def _expires_at(token: str) -> str:
    exp = _int_value(_jwt_claims(token).get("exp"), 0)
    if not exp:
        return ""
    return datetime.fromtimestamp(exp, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _jwt_claims(token: str) -> dict[str, Any]:
    parts = str(token or "").split(".")
    if len(parts) < 2:
        return {}
    try:
        payload = parts[1] + "=" * (-len(parts[1]) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload.encode("ascii")))
    except Exception:
        return {}
    return claims if isinstance(claims, dict) else {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _string_value(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _int_value(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
