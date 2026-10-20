"""Fail-closed transport policy for the on-host Local-LLM service.

Email content is sensitive and untrusted. It must never be sent through a
corporate proxy or to an arbitrary local/host-network service merely because a
dashboard or research CLI parameter was edited. This module deliberately has
no Streamlit dependency so every Local-LLM caller shares one policy.
"""

from __future__ import annotations

import ipaddress
import os
from typing import Any
from urllib.parse import SplitResult, urlsplit, urlunsplit

import requests


DEFAULT_LM_STUDIO_PORT = 1234


class LocalLLMEndpointError(ValueError):
    """Raised when a Local-LLM URL is outside the local-only trust boundary."""


def configured_lm_studio_port() -> int:
    """Read the single permitted Local-LLM port, failing closed if malformed."""

    raw = os.environ.get("AIC211_LM_STUDIO_PORT", str(DEFAULT_LM_STUDIO_PORT)).strip()
    try:
        port = int(raw)
    except ValueError as error:
        raise LocalLLMEndpointError("AIC211_LM_STUDIO_PORT must be a valid TCP port.") from error
    if not 1 <= port <= 65535:
        raise LocalLLMEndpointError("AIC211_LM_STUDIO_PORT must be between 1 and 65535.")
    return port


def _parse_endpoint(value: object) -> tuple[SplitResult, int]:
    try:
        parsed = urlsplit(str(value or ""))
        port = parsed.port
    except ValueError as error:
        raise LocalLLMEndpointError("Local-LLM endpoint has an invalid port.") from error
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or port is None
    ):
        raise LocalLLMEndpointError("Local-LLM endpoint must be an absolute HTTP(S) URL without credentials.")
    if port != configured_lm_studio_port():
        raise LocalLLMEndpointError("Local-LLM endpoint must use the one configured Local-LLM port.")
    return parsed, port


def _canonical_netloc(hostname: str, port: int) -> str:
    host = hostname.casefold()
    if host == "localhost":
        # Do not let a mutable hosts/DNS setting redirect a sensitive request.
        return f"127.0.0.1:{port}"
    try:
        address = ipaddress.ip_address(host)
    except ValueError as error:
        raise LocalLLMEndpointError("Local-LLM endpoint must use localhost or a literal loopback IP.") from error
    if not address.is_loopback:
        raise LocalLLMEndpointError("Local-LLM endpoint must use a literal loopback IP.")
    rendered = f"[{address.compressed}]" if address.version == 6 else address.compressed
    return f"{rendered}:{port}"


def canonical_local_llm_endpoint(value: object, *, require_base_url: bool = False) -> str:
    """Validate and canonicalise a local endpoint without performing DNS.

    ``localhost`` becomes ``127.0.0.1`` before transport. Host-network aliases
    such as ``host.docker.internal`` and non-loopback hostnames are rejected.
    A dashboard base URL cannot carry a path, query, or fragment.
    """

    parsed, port = _parse_endpoint(value)
    if require_base_url and (parsed.path not in {"", "/"} or parsed.query or parsed.fragment):
        raise LocalLLMEndpointError("Local-LLM base URL cannot include a path, query, or fragment.")
    return urlunsplit(
        (parsed.scheme, _canonical_netloc(parsed.hostname or "", port), parsed.path, parsed.query, parsed.fragment)
    )


def is_safe_local_llm_endpoint(value: object, *, require_base_url: bool = False) -> bool:
    """Boolean policy wrapper for UI validation and argument checking."""

    try:
        canonical_local_llm_endpoint(value, require_base_url=require_base_url)
    except LocalLLMEndpointError:
        return False
    return True


def request_local_llm(method: str, endpoint: object, *, timeout: float, **kwargs: Any) -> requests.Response:
    """Make a proxy-free request after revalidating the complete URL.

    ``Session.trust_env = False`` also blocks ``ALL_PROXY`` and related
    environment variables from carrying email content outside the machine.
    """

    target = canonical_local_llm_endpoint(endpoint)
    with requests.Session() as session:
        session.trust_env = False
        return session.request(method=method, url=target, timeout=timeout, **kwargs)
