from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any

from hr_agent_contracts import ActorContext


class AuthenticationError(PermissionError):
    pass


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


class IdentityProvider:
    """HMAC-signed compact identity token for the self-hosted P0 API boundary."""

    def __init__(
        self, secret: str | None, *, allow_demo: bool,
        default_company_id: str, default_operator_id: str, default_roles: list[str],
    ) -> None:
        self.secret = secret
        self.allow_demo = allow_demo
        self.default_company_id = default_company_id
        self.default_operator_id = default_operator_id
        self.default_roles = default_roles

    def resolve(
        self, authorization: str | None, company_id: str | None,
        operator_id: str | None, roles: str | None, request_id: str | None,
    ) -> ActorContext:
        if self.secret:
            if not authorization or not authorization.startswith("Bearer "):
                raise AuthenticationError("缺少 Bearer 身份令牌")
            payload = self.verify(authorization.removeprefix("Bearer ").strip())
            return ActorContext(
                company_id=str(payload["company_id"]),
                operator_id=str(payload["operator_id"]),
                roles=[str(item) for item in payload.get("roles", [])],
                request_id=request_id,
            )
        if not self.allow_demo:
            raise AuthenticationError("服务未配置身份密钥，且演示身份已关闭")
        return ActorContext(
            company_id=company_id or self.default_company_id,
            operator_id=operator_id or self.default_operator_id,
            roles=[item.strip() for item in (roles or ",".join(self.default_roles)).split(",") if item.strip()],
            request_id=request_id,
        )

    def verify(self, token: str) -> dict[str, Any]:
        if not self.secret:
            raise AuthenticationError("身份签名未配置")
        try:
            encoded_payload, encoded_signature = token.split(".", 1)
            expected = hmac.new(
                self.secret.encode("utf-8"), encoded_payload.encode("ascii"), hashlib.sha256,
            ).digest()
            if not hmac.compare_digest(expected, _b64decode(encoded_signature)):
                raise AuthenticationError("身份令牌签名无效")
            payload = json.loads(_b64decode(encoded_payload))
        except AuthenticationError:
            raise
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            raise AuthenticationError("身份令牌格式无效") from exc
        if float(payload.get("exp", 0)) <= time.time():
            raise AuthenticationError("身份令牌已过期")
        if not payload.get("company_id") or not payload.get("operator_id"):
            raise AuthenticationError("身份令牌缺少企业或操作者")
        return payload

    def issue_for_testing(self, payload: dict[str, Any]) -> str:
        if not self.secret:
            raise AuthenticationError("身份签名未配置")
        encoded = _b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        signature = hmac.new(
            self.secret.encode("utf-8"), encoded.encode("ascii"), hashlib.sha256,
        ).digest()
        return f"{encoded}.{_b64encode(signature)}"
