"""Versioned format registry with process limits; no network or external resources."""

import re
import subprocess
import sys
from dataclasses import dataclass, replace

from .models import PARSER_VERSION, ParsePolicy, ParseResult, canonical


@dataclass(frozen=True)
class ParserSpec:
    name: str
    version: str
    mimes: tuple[str, ...]
    max_bytes: int
    failures: tuple[str, ...] = (
        "numeric_markup_unsupported",
        "wrong_mime",
        "unsafe_document",
        "resource_limit",
        "parse_failed",
        "encrypted",
        "no_text_ocr_unsupported",
    )


REGISTRY = (
    ParserSpec("structured-xml", PARSER_VERSION, ("application/xml", "text/xml"), 2_000_000),
    ParserSpec(
        "structured-html", PARSER_VERSION, ("text/html", "application/xhtml+xml"), 2_000_000
    ),
    ParserSpec("trusted-pdf", PARSER_VERSION, ("application/pdf",), 2_000_000),
)


def parser_spec(mime, policy):
    spec = next((p for p in REGISTRY if mime in p.mimes), None)
    if spec and policy.version == "xml-jats-citations/1.0.0":
        return replace(spec, version="2.3.0")
    if spec and policy.version == "xml-jats/1.0.0":
        return replace(spec, version="2.2.0")
    if spec and policy.version == "xml-scripts/1.0.0":
        return replace(spec, version="2.1.0")
    return spec


def parse(
    data: bytes, mime: str, *, language: str = "und", policy: ParsePolicy | None = None
) -> ParseResult:
    policy = ParsePolicy.model_validate(dict(vars(policy or ParsePolicy())), strict=True)
    spec = parser_spec(mime, policy)
    if spec is None:
        return ParseResult(status="unsupported", reason="unsupported_mime")
    if policy.version in {
        "xml-scripts/1.0.0",
        "xml-jats/1.0.0",
        "xml-jats-citations/1.0.0",
    } and mime not in {
        "application/xml",
        "text/xml",
    }:
        return ParseResult(status="unsupported", reason="xml_scripts_mime")
    if not data:
        return ParseResult(status="failed", reason="empty_document")
    if len(data) > min(policy.max_bytes, spec.max_bytes):
        return ParseResult(status="failed", reason="resource_limit")
    if len(language) > 32:
        raise ValueError("language exceeds bound")
    import base64

    request = canonical(
        {
            "data": base64.b64encode(data).decode(),
            "mime": mime,
            "language": language,
            "policy": policy.model_dump(mode="json"),
        }
    )
    try:
        result = subprocess.run(
            [sys.executable, "-m", "packages.knowledge_index.worker"],
            input=request,
            capture_output=True,
            timeout=policy.timeout_seconds + 2,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ParseResult(status="failed", reason="resource_limit")
    if result.returncode != 0 or len(result.stdout) > 4_000_000:
        return ParseResult(status="failed", reason="resource_limit")
    return ParseResult.model_validate_json(result.stdout)


def fts_text(text: str) -> str:
    # Versioned minimal multilingual baseline: CJK unigram tokens plus normal words.
    return re.sub(r"([\u3400-\u9fff])", r" \1 ", " ".join(text.split()))
