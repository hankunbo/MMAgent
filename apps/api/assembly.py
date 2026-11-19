"""Explicit local dependency installation; no credential discovery or startup I/O."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import Field

from apps.api.auth import LocalIdentities
from apps.api.models import Value


class Baseline(Value):
    version: Literal["research-http-baseline/1.0.0"]
    baseline_only: Literal[True]
    max_request_bytes: int = Field(ge=1024, le=1048576)
    max_artifact_bytes: int = Field(ge=1, le=4194304)
    max_response_bytes: int = Field(ge=1024, le=16777216)
    default_page_size: int = Field(ge=1, le=100)
    max_page_size: int = Field(ge=1, le=100)
    allowed_media_types: tuple[Literal["image/png", "image/jpeg"], ...]
    live_domain_acceptance: Literal["live_domain_acceptance_not_enabled"]

    @classmethod
    def load(cls):
        path = Path(__file__).resolve().parents[2] / "configs/api/research_baseline_v1.json"
        return cls.model_validate(json.loads(path.read_text()))


@dataclass(frozen=True)
class CaseSource:
    sample: object
    dataset_code: str
    checksum: str


@dataclass(frozen=True)
class ArtifactSource:
    artifact: object
    image_binding: object
    byte_size: int


@dataclass(frozen=True)
class ExtensionBinding:
    owner: object
    stage_key: str
    worker: str


@dataclass(frozen=True)
class Dependencies:
    identities: LocalIdentities
    runtime: object = None
    research: object = None
    preparation: object = None
    visible: object = None
    objects: object = None
    case_sources: dict = field(default_factory=dict)
    artifact_sources: dict = field(default_factory=dict)
    planners: dict = field(default_factory=dict)
    executions: dict = field(default_factory=dict)
    reports: dict = field(default_factory=dict)
    graphs: dict = field(default_factory=dict)
    validations: dict = field(default_factory=dict)
    counterfactuals: dict = field(default_factory=dict)
    expert_history: dict = field(default_factory=dict)
    expert_requests: dict = field(default_factory=dict)
    offline_boards: dict = field(default_factory=dict)
    gaps: object = None
    discovery: object = None
    adapter: object = None
    extensions: dict = field(default_factory=dict)
    audit_reader: object = None
    audit_writer: object = None
    dispatch: object = None
