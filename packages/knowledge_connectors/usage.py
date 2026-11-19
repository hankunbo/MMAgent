"""Versioned descriptive tool usage; no medical authorization."""

import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from packages.contracts import VersionedReference
from packages.knowledge_connectors.ports import digest

Text = Annotated[str, StringConstraints(min_length=1, max_length=4000)]
Items = Annotated[tuple[Text, ...], Field(min_length=1, max_length=20)]


class ToolUsageProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    schema_version: Literal["knowledge-tool-usage/1.0.0"] = "knowledge-tool-usage/1.0.0"
    profile_version: Annotated[str, StringConstraints(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")] = (
        "1.0.0"
    )
    provider_ref: VersionedReference
    key: Annotated[str, StringConstraints(pattern=r"^[a-z_]{2,32}$")]
    name: Text
    purpose: Text
    best_for: Items
    not_for: Items
    inputs: Items
    outputs: Items
    workflow: Items
    examples: Items
    prerequisites: Items
    limitations: Items
    failure_actions: Items
    checksum: Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]

    @model_validator(mode="after")
    def valid_digest(self):
        raw = self.model_dump(mode="json", exclude={"checksum"})
        expected = digest(
            json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        )
        if self.checksum != expected:
            raise ValueError("tool_usage_checksum_mismatch")
        if not self.provider_ref.code.startswith("SOURCE:PROVIDER:"):
            raise ValueError("tool_usage_requires_provider")
        return self


def seal_profile(**fields):
    raw = {"schema_version": "knowledge-tool-usage/1.0.0", "profile_version": "1.0.0", **fields}
    raw["checksum"] = digest(
        json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    )
    return ToolUsageProfile.model_validate_json(json.dumps(raw, ensure_ascii=False))
