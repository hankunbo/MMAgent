"""Stable controlled vocabulary codes and opaque entity identifiers.

Database surrogate keys are deliberately outside this module. The validation here
is syntactic: an ``ANON_`` marker records an upstream anonymization claim but cannot
itself prove that source data contains no identifying information.
"""

from __future__ import annotations

import re
import secrets
from enum import StrEnum


class IdentifierError(ValueError):
    """Raised when a stable business code is malformed or in the wrong domain."""


class CodeNamespace(StrEnum):
    CONCEPT = "CONCEPT"
    CLAIM = "CLAIM"
    EVIDENCE = "EVIDENCE"
    GOAL = "GOAL"
    ARTIFACT = "ARTIFACT"
    CAPABILITY = "CAPABILITY"
    TOOL = "TOOL"
    IMPLEMENTATION = "IMPLEMENTATION"
    WORKFLOW = "WORKFLOW"
    CASE = "CASE"
    ENCOUNTER = "ENCOUNTER"
    GRAPH = "GRAPH"
    MEMORY = "MEMORY"
    MESSAGE = "MESSAGE"
    RUN = "RUN"
    STEP = "STEP"
    TASK = "TASK"
    POLICY = "POLICY"
    RULE = "RULE"
    TEMPLATE = "TEMPLATE"
    SOURCE = "SOURCE"
    ASSET = "ASSET"
    BINDING = "BINDING"
    VALIDATION = "VALIDATION"
    EPISODE = "EPISODE"
    CFPAIR = "CFPAIR"
    REQUIREMENT = "REQUIREMENT"
    EXPERT_ASSESSMENT = "EXPERT_ASSESSMENT"


class StableCodeKind(StrEnum):
    CONTROLLED_TERM = "CONTROLLED_TERM"
    OPAQUE_ENTITY_ID = "OPAQUE_ENTITY_ID"


CONTROLLED_TERM_NAMESPACES = frozenset(
    {
        CodeNamespace.CONCEPT,
        CodeNamespace.CLAIM,
        CodeNamespace.EVIDENCE,
        CodeNamespace.GOAL,
        CodeNamespace.ARTIFACT,
        CodeNamespace.CAPABILITY,
        CodeNamespace.TOOL,
        CodeNamespace.WORKFLOW,
        CodeNamespace.POLICY,
        CodeNamespace.RULE,
        CodeNamespace.TEMPLATE,
        CodeNamespace.SOURCE,
    }
)

OPAQUE_ENTITY_NAMESPACES = frozenset(set(CodeNamespace) - CONTROLLED_TERM_NAMESPACES)

_SEGMENT_RE = re.compile(r"^[A-Z0-9][A-Z0-9._-]{0,63}$")
_STABLE_CODE_RE = re.compile(r"^[A-Z][A-Z0-9_]{1,31}(?::[A-Z0-9][A-Z0-9._-]{0,63})+$")
_NORMALIZE_RE = re.compile(r"[^A-Z0-9._-]+")
_OPAQUE_TOKEN_RE = re.compile(r"^(?:ANON|OID)_[A-Z0-9][A-Z0-9_-]{5,63}$")
_URI_PREFIX_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_SCHEME_OR_DRIVE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


def _reject_path_or_uri(value: str, *, field: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if (
        _URI_PREFIX_RE.match(value)
        or value.startswith(("/", "~", "./", "../"))
        or "/" in value
        or "\\" in value
        or "\x00" in value
    ):
        raise IdentifierError(f"{field} must not be a path or URI")


def _parse_namespace(value: str) -> CodeNamespace:
    _reject_path_or_uri(value, field="stable code")
    if not _STABLE_CODE_RE.fullmatch(value):
        raise IdentifierError(
            "stable code must use canonical NAMESPACE:SEGMENT syntax with uppercase ASCII"
        )
    namespace_text = value.partition(":")[0]
    try:
        return CodeNamespace(namespace_text)
    except ValueError as exc:
        raise IdentifierError(f"unknown stable code namespace: {namespace_text}") from exc


class StableCode(str):
    """Base type that dispatches to a controlled term or opaque entity identifier."""

    def __new__(cls, value: str) -> StableCode:
        if not isinstance(value, str):
            raise TypeError("stable code must be a string")
        namespace = _parse_namespace(value)
        if cls is StableCode:
            concrete_type = (
                ControlledTermCode if namespace in CONTROLLED_TERM_NAMESPACES else OpaqueEntityId
            )
            return concrete_type(value)
        return str.__new__(cls, value)

    @property
    def namespace(self) -> CodeNamespace:
        return CodeNamespace(self.partition(":")[0])

    @property
    def parts(self) -> tuple[str, ...]:
        return tuple(self.split(":")[1:])

    @property
    def kind(self) -> StableCodeKind:
        if isinstance(self, ControlledTermCode):
            return StableCodeKind.CONTROLLED_TERM
        return StableCodeKind.OPAQUE_ENTITY_ID


class ControlledTermCode(StableCode):
    """A stable label whose namespace belongs to a controlled vocabulary."""

    def __new__(cls, value: str) -> ControlledTermCode:
        code = super().__new__(cls, value)
        if code.namespace not in CONTROLLED_TERM_NAMESPACES:
            raise IdentifierError(f"{code.namespace.value} is an opaque entity namespace")
        return code


class OpaqueEntityId(StableCode):
    """An entity identifier with one explicitly opaque or anonymized token."""

    def __new__(cls, value: str) -> OpaqueEntityId:
        code = super().__new__(cls, value)
        if code.namespace not in OPAQUE_ENTITY_NAMESPACES:
            raise IdentifierError(f"{code.namespace.value} is a controlled-term namespace")
        if len(code.parts) != 1 or not _OPAQUE_TOKEN_RE.fullmatch(code.parts[0]):
            raise IdentifierError(
                "opaque entity ID requires one ANON_ or OID_ token; "
                "raw entity identifiers are forbidden"
            )
        return code


def normalize_code_segment(value: str) -> str:
    """Convert a non-sensitive controlled label to one deterministic code segment."""

    _reject_path_or_uri(value, field="controlled label")
    if _SCHEME_OR_DRIVE_RE.match(value):
        raise IdentifierError("controlled label must not be a path or URI")
    normalized = _NORMALIZE_RE.sub("_", value.strip().upper()).strip("_")
    if not normalized or not _SEGMENT_RE.fullmatch(normalized):
        raise IdentifierError(f"cannot normalize controlled label: {value!r}")
    return normalized


def make_controlled_term_code(namespace: CodeNamespace, *labels: str) -> ControlledTermCode:
    """Build a controlled vocabulary code from one or more non-sensitive labels."""

    if not isinstance(namespace, CodeNamespace):
        raise TypeError("namespace must be a CodeNamespace")
    if namespace not in CONTROLLED_TERM_NAMESPACES:
        raise IdentifierError(f"{namespace.value} requires an opaque entity ID")
    if not labels:
        raise IdentifierError("controlled term code requires at least one label")
    return ControlledTermCode(
        ":".join((namespace.value, *(normalize_code_segment(label) for label in labels)))
    )


def make_stable_code(namespace: CodeNamespace, *labels: str) -> ControlledTermCode:
    """Backward-compatible name for controlled term construction only."""

    return make_controlled_term_code(namespace, *labels)


def make_opaque_entity_id(namespace: CodeNamespace, opaque_token: str) -> OpaqueEntityId:
    """Build an entity ID from an upstream anonymized or generated opaque token."""

    if not isinstance(namespace, CodeNamespace):
        raise TypeError("namespace must be a CodeNamespace")
    if namespace not in OPAQUE_ENTITY_NAMESPACES:
        raise IdentifierError(f"{namespace.value} requires a controlled term code")
    _reject_path_or_uri(opaque_token, field="opaque token")
    return OpaqueEntityId(f"{namespace.value}:{opaque_token}")


def generate_opaque_entity_id(namespace: CodeNamespace) -> OpaqueEntityId:
    """Generate a non-semantic identifier without accepting source entity text."""

    return make_opaque_entity_id(namespace, f"OID_{secrets.token_hex(16).upper()}")


def validate_stable_code(
    value: str,
    expected_namespace: CodeNamespace | None = None,
    expected_kind: StableCodeKind | None = None,
) -> StableCode:
    """Validate a code and optionally enforce its namespace and semantic kind."""

    code = StableCode(value)
    if expected_namespace is not None and code.namespace is not expected_namespace:
        raise IdentifierError(
            f"expected namespace {expected_namespace.value}, got {code.namespace.value}"
        )
    if expected_kind is not None and code.kind is not expected_kind:
        raise IdentifierError(f"expected kind {expected_kind.value}, got {code.kind.value}")
    return code


def validate_controlled_term_code(value: str) -> ControlledTermCode:
    """Validate and return a controlled vocabulary label."""

    return ControlledTermCode(value)


def validate_opaque_entity_id(value: str) -> OpaqueEntityId:
    """Validate the syntax and anonymization marker of an opaque entity ID."""

    return OpaqueEntityId(value)
