"""Typed public bibliographic metadata, never a patient-input or archive port.

Python normalizes before sealing. SQL validates the identical canonical wire
form: it never silently rewrites an immutable record/checksum supplied by SQL.
"""

import json
import re
from pathlib import Path
from typing import Literal

from packages.knowledge_connectors.ports import StrictModel, digest

from .privacy import PrivacyError, canonical_doi, safe_text
from .provider import discovery_url

POLICY = json.loads(Path(__file__).with_name("discovery_policy.json").read_text())
CHECKSUM = digest(
    json.dumps(POLICY, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
)


class DiscoveryPolicy(StrictModel):
    version: Literal["discovery-metadata/1.0.0"] = "discovery-metadata/1.0.0"
    checksum: Literal[CHECKSUM] = CHECKSUM
    provenance: Literal["configured_provider_bibliography"] = "configured_provider_bibliography"


def title(value):
    value = safe_text(value)
    if len(value) > POLICY["max_title_length"]:
        raise PrivacyError()
    return value


def identifier(kind, value, proof):
    if kind == "URL":
        value = discovery_url(value, proof)
        if len(value) > POLICY["max_url_length"]:
            raise PrivacyError()
        return value
    rule = POLICY["identifiers"].get(kind)
    if rule is None:
        raise PrivacyError()
    value = canonical_doi(value) if kind == "DOI" else safe_text(value)
    if len(value) > rule["max_length"] or not re.fullmatch(rule["pattern"], value):
        raise PrivacyError()
    return value


def record_id(value, proof):
    if value is None:
        return None
    value = canonical_doi(value) if proof.adapter == "crossref" else safe_text(value)
    if len(value) > 512 or not re.fullmatch(POLICY["record_ids"][proof.adapter], value):
        raise PrivacyError()
    return value


def validate_candidate(hit):
    if title(hit.title) != hit.title:
        raise PrivacyError()
    m = hit.metadata
    if m.scope() != m.configuration.scope():
        raise PrivacyError()
    if record_id(m.provider_record_id, m.configuration) != m.provider_record_id:
        raise PrivacyError()
    for item in (*hit.identifiers, *m.discovery_identifiers):
        if identifier(item.identifier_type, item.value, m.configuration) != item.value:
            raise PrivacyError()
    return hit
