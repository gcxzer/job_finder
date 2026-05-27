from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class CodexCredentials:
    access_token: str = field(default="", repr=False)
    refresh_token: str = field(default="", repr=False)
    id_token: str = field(default="", repr=False)
    account_id: str = ""
    account_email: str = ""
    plan_type: str = ""
    expires_at: str = ""
    last_refresh: str = ""
    source: str = ""
    base_url: str = ""

    @property
    def configured(self) -> bool:
        return bool(self.access_token or self.refresh_token)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CodexCredentials":
        payload = data or {}
        return cls(
            access_token=str(payload.get("accessToken") or payload.get("access_token") or ""),
            refresh_token=str(payload.get("refreshToken") or payload.get("refresh_token") or ""),
            id_token=str(payload.get("idToken") or payload.get("id_token") or ""),
            account_id=str(payload.get("accountId") or payload.get("account_id") or ""),
            account_email=str(payload.get("accountEmail") or payload.get("account_email") or ""),
            plan_type=str(payload.get("planType") or payload.get("plan_type") or ""),
            expires_at=str(payload.get("expiresAt") or payload.get("expires_at") or ""),
            last_refresh=str(payload.get("lastRefresh") or payload.get("last_refresh") or ""),
            source=str(payload.get("source") or ""),
            base_url=str(payload.get("baseUrl") or payload.get("base_url") or ""),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "provider": "codex-oauth",
            "authMode": "chatgpt",
            "accessToken": self.access_token,
            "refreshToken": self.refresh_token,
            "idToken": self.id_token,
            "accountId": self.account_id,
            "accountEmail": self.account_email,
            "planType": self.plan_type,
            "expiresAt": self.expires_at,
            "lastRefresh": self.last_refresh,
            "source": self.source,
            "baseUrl": self.base_url,
        }

