"""Opaque local credentials map to trusted, immutable scope and role bindings."""

import hashlib
import re
from dataclasses import dataclass
from typing import Literal

from apps.api.errors import Fault


@dataclass(frozen=True)
class Principal:
    actor_code: str
    role: Literal[
        "researcher",
        "read_only",
        "restricted_agent",
        "offline_reviewer",
        "clinical_reviewer",
        "tool_admin",
    ]
    case_codes: frozenset[str] = frozenset()
    dataset_codes: frozenset[str] = frozenset()
    synthetic: bool = True

    def __post_init__(self):
        roles = {
            "researcher",
            "read_only",
            "restricted_agent",
            "offline_reviewer",
            "clinical_reviewer",
            "tool_admin",
        }

        def code(value):
            return isinstance(value, str) and re.fullmatch(
                r"[A-Z][A-Z0-9_]*:[A-Za-z0-9_.:-]+", value
            )

        if (
            not isinstance(self.role, str)
            or self.role not in roles
            or not code(self.actor_code)
            or type(self.synthetic) is not bool
            or type(self.case_codes) is not frozenset
            or type(self.dataset_codes) is not frozenset
            or not all(code(v) and v.startswith("CASE:") for v in self.case_codes)
            or not all(code(v) for v in self.dataset_codes)
        ):
            raise ValueError("explicit_local_identity_required")


class LocalIdentities:
    def __init__(self, bindings):
        if any(
            not isinstance(k, str) or not k or not isinstance(v, Principal)
            for k, v in bindings.items()
        ):
            raise ValueError("explicit_local_identity_required")
        self._bindings = {hashlib.sha256(k.encode()).digest(): v for k, v in bindings.items()}

    def authenticate(self, authorization):
        if not authorization or not authorization.startswith("Bearer "):
            raise Fault("AUTHENTICATION_REQUIRED")
        p = self._bindings.get(hashlib.sha256(authorization[7:].encode()).digest())
        if p is None:
            raise Fault("AUTHENTICATION_REQUIRED")
        return p


def require_role(principal, *roles):
    role = "offline_reviewer" if principal.role == "clinical_reviewer" else principal.role
    if role not in roles:
        raise Fault("SCOPE_FORBIDDEN")
