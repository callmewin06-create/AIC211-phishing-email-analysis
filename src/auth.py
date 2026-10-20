"""Small, dependency-free authentication and role primitives.

The dashboard is often installed in a private network, but a private network
is not an authorisation boundary.  This module intentionally has no Streamlit
dependency so the same policy can be used by a future gateway/API service.

Configure identities with ``AIC211_AUTH_IDENTITIES_JSON``.  The value is a
JSON list of objects with ``subject``, ``token_env`` and ``roles`` keys, for
example::

    [{"subject":"soc-analyst-1", "token_env":"AIC211_ANALYST_TOKEN",
      "roles":["viewer", "analyst"]}]

The token itself stays in the named environment variable (or a secret manager
which injects it), never in a committed configuration file.
"""

from __future__ import annotations

from dataclasses import dataclass
import hmac
import json
import os
from typing import Iterable


class AuthenticationError(PermissionError):
    """Raised when an identity cannot be authenticated or authorised."""


ROLE_VIEWER = "viewer"
ROLE_ANALYST = "analyst"
ROLE_MODEL_OPERATOR = "model_operator"
ROLE_ADMIN = "admin"
ROLE_SYSTEM = "system"
VALID_ROLES = frozenset({ROLE_VIEWER, ROLE_ANALYST, ROLE_MODEL_OPERATOR, ROLE_ADMIN, ROLE_SYSTEM})


@dataclass(frozen=True)
class Actor:
    subject: str
    roles: frozenset[str]

    def has_any_role(self, *required: str) -> bool:
        return ROLE_ADMIN in self.roles or bool(self.roles.intersection(required))


@dataclass(frozen=True)
class _ConfiguredIdentity:
    subject: str
    token: str
    roles: frozenset[str]


def _configured_identities(raw_config: str | None = None) -> tuple[_ConfiguredIdentity, ...]:
    raw_config = raw_config if raw_config is not None else os.environ.get("AIC211_AUTH_IDENTITIES_JSON", "")
    if not raw_config.strip():
        return ()
    try:
        payload = json.loads(raw_config)
    except json.JSONDecodeError as error:
        raise AuthenticationError("AIC211_AUTH_IDENTITIES_JSON is not valid JSON") from error
    if not isinstance(payload, list):
        raise AuthenticationError("AIC211_AUTH_IDENTITIES_JSON must be a list")

    identities: list[_ConfiguredIdentity] = []
    for item in payload:
        if not isinstance(item, dict):
            raise AuthenticationError("Each configured identity must be an object")
        subject = str(item.get("subject", "")).strip()
        token_env = str(item.get("token_env", "")).strip()
        roles = frozenset(str(role).strip() for role in item.get("roles", ()) if str(role).strip())
        if not subject or not token_env or not roles or not roles.issubset(VALID_ROLES):
            raise AuthenticationError("Configured identity is incomplete or has an invalid role")
        token = os.environ.get(token_env, "")
        if not token:
            raise AuthenticationError(f"Configured token variable {token_env!r} is empty")
        identities.append(_ConfiguredIdentity(subject, token, roles))
    return tuple(identities)


def authentication_is_configured() -> bool:
    """Return whether at least one valid identity is configured.

    Configuration errors deliberately count as not configured for UI startup;
    callers can use :func:`authenticate` to show the precise error safely.
    """

    try:
        return bool(_configured_identities())
    except AuthenticationError:
        return False


def authenticate(token: str, configured: Iterable[_ConfiguredIdentity] | None = None) -> Actor:
    """Authenticate a bearer token using constant-time comparisons.

    No development or anonymous fallback exists here.  A caller that wants a
    test actor must construct it explicitly, making accidental production
    bypasses visible in code review.
    """

    supplied = str(token or "")
    if not supplied:
        raise AuthenticationError("Authentication token is required")
    identities = tuple(configured) if configured is not None else _configured_identities()
    for identity in identities:
        if hmac.compare_digest(supplied, identity.token):
            return Actor(identity.subject, identity.roles)
    raise AuthenticationError("Invalid authentication token")


def require_role(actor: Actor | None, *required: str) -> Actor:
    """Require an authenticated actor with at least one of ``required`` roles."""

    if actor is None:
        raise AuthenticationError("An authenticated identity is required")
    if not required or actor.has_any_role(*required):
        return actor
    raise AuthenticationError("The authenticated identity lacks the required role")
