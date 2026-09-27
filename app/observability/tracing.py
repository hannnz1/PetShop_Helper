"""Per-turn Langfuse callbacks with local-only transport and failure isolation."""

import hashlib
import hmac
import ipaddress
import logging
import re
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

from langchain_core.callbacks import BaseCallbackHandler

from app.config import Settings

logger = logging.getLogger(__name__)

_PRIVATE_V4 = tuple(ipaddress.ip_network(block) for block in (
    "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16",
))
_PRIVATE_V6 = ipaddress.ip_network("fc00::/7")
_initialized_keys: set[str] = set()


def _self_hosted_url(url: str) -> bool:
    """Allow loopback and literal private network hosts, never Cloud DNS names."""
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        if (parsed.scheme not in {"http", "https"} or not host or not parsed.netloc
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.port == 0):
            return False
        if host.lower().rstrip(".") == "localhost":
            return True
        address = ipaddress.ip_address(host)
    except (ValueError, TypeError):
        return False
    if address.is_loopback:
        return True
    if isinstance(address, ipaddress.IPv4Address):
        return any(address in block for block in _PRIVATE_V4)
    return address in _PRIVATE_V6


def turn_metadata(settings: Settings, user_id: str, conversation_id: int,
                  turn_id: str) -> dict[str, str]:
    """Use a keyed stable pseudonym and only non-secret correlation values."""
    metadata = {
        "turn_id": turn_id,
        "langfuse_session_id": str(conversation_id),
        "langfuse_trace_name": "petshop-graph-turn",
    }
    if settings.langfuse_secret_key is not None:
        digest = hmac.new(settings.langfuse_secret_key.get_secret_value().encode(),
                          user_id.encode(), hashlib.sha256).hexdigest()
        metadata["langfuse_user_id"] = f"user-{digest}"
    return metadata


def make_trace_mask(settings: Settings) -> Callable[[Any], Any]:
    """Redact configured credentials and private headers from SDK-owned payloads."""
    candidates = [settings.chat_api_key, settings.siliconflow_api_key,
                  settings.rerank_api_key, settings.langfuse_public_key,
                  settings.langfuse_secret_key, settings.observability_admin_token]
    values = [item.get_secret_value() for item in candidates if item is not None]
    values += [settings.database_url, settings.test_database_url]
    secrets = sorted({value for value in values if value}, key=len, reverse=True)

    def redact(value: Any) -> Any:
        if isinstance(value, str):
            for secret in secrets:
                value = value.replace(secret, "[REDACTED]")
            return re.sub(r"(?i)(authorization\s*[:=]\s*)(?:bearer\s+)?\S+",
                          r"\1[REDACTED]", value)
        if isinstance(value, dict):
            return {key: "[REDACTED]" if str(key).lower() in {
                "authorization", "proxy-authorization", "x-api-key",
            } else redact(item) for key, item in value.items()}
        if isinstance(value, list):
            return [redact(item) for item in value]
        if isinstance(value, tuple):
            return tuple(redact(item) for item in value)
        return value

    def mask(*, data: Any, **kwargs: Any) -> Any:
        return redact(data)

    return mask


def _new_handler(*, public_key: str, secret_key: str, base_url: str,
                 mask: Callable[[Any], Any]) -> BaseCallbackHandler:
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler

    # SDK v4 shares resources by public key; CallbackHandler looks up this client.
    Langfuse(public_key=public_key, secret_key=secret_key, base_url=base_url, mask=mask)
    _initialized_keys.add(public_key)
    return CallbackHandler(public_key=public_key)


def _get_client(*, public_key: str):
    from langfuse import get_client
    return get_client(public_key=public_key)


def close_tracing(settings: Settings) -> None:
    """Flush the SDK at application shutdown without affecting resource cleanup."""
    public = settings.langfuse_public_key
    if not settings.langfuse_enabled or public is None:
        return
    key = public.get_secret_value()
    if key not in _initialized_keys:
        return
    try:
        _get_client(public_key=key).shutdown()
    except Exception as exc:
        logger.warning("Langfuse shutdown failed (%s)", type(exc).__name__)
    finally:
        _initialized_keys.discard(key)


class _SafeCallbackHandler(BaseCallbackHandler):
    """Prevent telemetry hooks from changing Graph or SSE results."""

    raise_error = False

    def __init__(self, delegate: BaseCallbackHandler) -> None:
        self._delegate = delegate

    def _forward(self, event_name: str, *args, **kwargs):
        try:
            return getattr(self._delegate, event_name)(*args, **kwargs)
        except Exception as exc:
            # SDK error text can contain request details or credentials.
            logger.warning("Langfuse callback %s failed (%s)", event_name, type(exc).__name__)
            return None


def _safe_event(name: str):
    def event(self, *args, **kwargs):
        return self._forward(name, *args, **kwargs)
    return event


for _event_name in (
    "on_agent_action", "on_agent_finish", "on_chain_start", "on_chain_end",
    "on_chain_error", "on_chat_model_start", "on_llm_start", "on_llm_end",
    "on_llm_error", "on_llm_new_token", "on_retriever_start", "on_retriever_end",
    "on_retriever_error", "on_tool_start", "on_tool_end", "on_tool_error",
):
    setattr(_SafeCallbackHandler, _event_name, _safe_event(_event_name))


def make_turn_callbacks(settings: Settings, user_id: str, conversation_id: int,
                        turn_id: str) -> list[BaseCallbackHandler]:
    """Create a callback only with complete local self-hosted configuration."""
    if not settings.langfuse_enabled:
        return []
    public = settings.langfuse_public_key
    secret = settings.langfuse_secret_key
    if public is None or secret is None:
        logger.warning("Langfuse enabled without both credentials; tracing skipped")
        return []
    public_key = public.get_secret_value()
    secret_key = secret.get_secret_value()
    if not public_key.strip() or not secret_key.strip():
        logger.warning("Langfuse enabled with blank credentials; tracing skipped")
        return []
    if not _self_hosted_url(settings.langfuse_base_url):
        logger.warning("Langfuse URL is not an allowed local/private endpoint; tracing skipped")
        return []
    try:
        handler = _new_handler(public_key=public_key,
                               secret_key=secret_key,
                               base_url=settings.langfuse_base_url,
                               mask=make_trace_mask(settings))
    except Exception as exc:
        logger.warning("Langfuse initialization failed (%s)", type(exc).__name__)
        return []
    return [_SafeCallbackHandler(handler)]
