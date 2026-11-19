"""Versioned, default-deny input vocabulary and defensive discovery text checks.

Regexes are defense in depth, never proof of de-identification. Input terms must
resolve in the shipped, reviewed general-knowledge catalog; runtime callers cannot
register terms. Case abstraction belongs upstream of this package.
"""

import html
import json
import re
import unicodedata
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

from packages.knowledge_connectors.ports import digest


class PrivacyError(ValueError):
    def __init__(self):
        super().__init__("search privacy policy rejected metadata")


POLICY_VERSION = "patient-free/3.0.0"
CATALOG = json.loads(Path(__file__).with_name("terminology.json").read_text())
POLICY_CHECKSUM = digest(
    json.dumps(CATALOG, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
)
FORBIDDEN = re.compile(
    r"https?\s*:\s*/|@|\b(?:tool|email|mailto|api[_ -]?key|authorization)\s*[:=]"
    r"|\bbearer\s|\bsk-[A-Za-z0-9_-]{8,}"
    r"|\b(?:mrn|dob|date of birth|medical record|hospital number|case number"
    r"|patient id|subject id|social security|phone|telephone|home address)\b"
    r"|\b(?:patient|subject)\s+(?:name|[A-Z][a-z]+\s+[A-Z][a-z]+)\b"
    r"|\b(?:current|this|our)\s+(?:case|patient|subject)\b"
    r"|\b(?:patient|he|she)\s+(?:reports|presents|complains|was admitted)\b"
    r"|\b(?:visit date|admission date)\b|病历号|住院号|身份证|出生日期"
    r"|患者姓名|患者编号|受试者编号|就诊日期|本例|该患者|患者自述",
    re.I,
)


def safe_text(value: str) -> str:
    if type(value) is not str or not 0 < len(value) <= 8192:
        raise PrivacyError()
    current = value
    for _ in range(16):
        normalized = unicodedata.normalize("NFKC", current)
        if FORBIDDEN.search(normalized) or any(
            unicodedata.category(c).startswith("C") for c in normalized
        ):
            raise PrivacyError()
        try:
            decoded = html.unescape(unquote(normalized, errors="strict"))
        except (ValueError, UnicodeError):
            raise PrivacyError() from None
        if len(decoded) > 8192 or len(decoded) > max(256, len(value) * 4):
            raise PrivacyError()
        if decoded == current:
            if re.search(r"%[a-z0-9%]|&#?|\\(?:u|x)[0-9a-f]", decoded, re.I):
                # Ordinary ampersands in medical titles are allowed, entity syntax is not.
                if re.search(r"%[a-z0-9%]|&(?:#|[a-z]+;)|\\(?:u|x)[0-9a-f]", decoded, re.I):
                    raise PrivacyError()
            return normalized
        current = decoded
    raise PrivacyError()


URL_POLICY_VERSION = "discovery-url/1.0.0"
URL_HOSTS = frozenset(
    {
        "doi.org",
        "dx.doi.org",
        "pubmed.ncbi.nlm.nih.gov",
        "europepmc.org",
        "www.nice.org.uk",
        "www.aao.org",
    }
)


def single_path(value: str) -> str:
    """Decode exactly once; a second decoding must never change meaning."""
    if re.search(r"%(?![0-9a-fA-F]{2})", value):
        raise PrivacyError()
    try:
        decoded = unicodedata.normalize("NFKC", unquote(value, errors="strict"))
    except (ValueError, UnicodeError):
        raise PrivacyError() from None
    if any(c in decoded for c in "%\\?#") or html.unescape(decoded) != decoded:
        raise PrivacyError()
    if "//" in decoded or any(p in (".", "..") for p in decoded.split("/")):
        raise PrivacyError()
    safe_text(decoded)
    return decoded


def safe_url(value: str, *, allowed_hosts: tuple[str, ...] | None = None) -> str:
    """Public discovery URL policy, shared by all connector result consumers."""
    try:
        if type(value) is not str or not 0 < len(value) <= 2048:
            raise PrivacyError()
        if any(unicodedata.category(c).startswith("C") for c in value):
            raise PrivacyError()
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https"
            or parsed.hostname not in (URL_HOSTS if allowed_hosts is None else allowed_hosts)
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port not in (None, 443)
            or "?" in value
            or "#" in value
            or not parsed.netloc.isascii()
        ):
            raise PrivacyError()
        path = single_path(parsed.path or "/")
        return "https://" + parsed.hostname + quote(path, safe="/():;,+-._~!$'*= ".replace(" ", ""))
    except Exception:
        raise PrivacyError() from None


IDENTIFIER_POLICY = json.loads(Path(__file__).with_name("identifier_policy.json").read_text())
IDENTIFIER_CHECKSUM = digest(
    json.dumps(
        IDENTIFIER_POLICY, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
)


def canonical_doi(value: str) -> str:
    """Syntax admission only; no claim of existence or patient de-identification."""
    if type(value) is not str or not 0 < len(value) <= 1024:
        raise PrivacyError()
    if value.lower().startswith("https://"):
        normalized = safe_url(value)
        parsed = urlsplit(normalized)
        if parsed.hostname not in ("doi.org", "dx.doi.org"):
            raise PrivacyError()
        value = parsed.path[1:]
    elif value.lower().startswith("doi:"):
        value = value[4:]
    value = single_path(value).casefold()
    if len(value) > IDENTIFIER_POLICY["max_length"] or not re.fullmatch(
        IDENTIFIER_POLICY["canonical_pattern"], value
    ):
        raise PrivacyError()
    return value


def controlled(category: str, value: str) -> str:
    term = " ".join(safe_text(value).split()).casefold()
    if term not in CATALOG[category]:
        raise PrivacyError()
    return term


def coded(code: str, terms: tuple[str, ...]) -> None:
    allowed = CATALOG["concepts"].get(code)
    if allowed is None or any(term not in allowed for term in terms):
        raise PrivacyError()
