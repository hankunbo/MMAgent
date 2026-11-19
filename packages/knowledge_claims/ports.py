"""Configured structured extraction and explicit bounded offline batch execution."""

import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal, Protocol
from urllib.parse import urlsplit

from pydantic import Field, ValidationError, model_validator

from packages.knowledge_index.models import canonical, digest
from packages.object_store import checksum

from .evidence import Rejected
from .models import BatchPolicy, Model, StructuredOutput, Term, Version

ROOT = Path(__file__).resolve().parents[2]
PROMPT = ROOT / "configs/prompts/claim_extraction_v2.txt"


class LLMConfig(Model):
    provider: Literal["openai_compatible", "openai"]
    api_key_env: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")
    base_url: str = Field(max_length=2048)
    model: Term
    endpoint: Literal["chat_completions", "responses"]
    timeout_seconds: int = Field(ge=1, le=300)
    max_retries: int = Field(ge=0, le=3)
    store: bool

    @model_validator(mode="after")
    def safe_endpoint(self):
        url = urlsplit(self.base_url)
        if (
            url.scheme != "https"
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
            or self.store is not False
            or (self.provider == "openai_compatible" and self.endpoint != "chat_completions")
            or (
                self.provider == "openai"
                and self.base_url.rstrip("/") != "https://api.openai.com/v1"
            )
        ):
            raise ValueError("invalid nonsecret LLM configuration")
        return self


def load_llm_config(path=ROOT / "configs/environments/default.yaml", *, environ=None):
    """Read the project's existing scalar llm YAML block; refuse unsupported syntax."""
    env = os.environ if environ is None else environ
    if path.stat().st_size > 65536:
        raise ValueError("configuration_limit")
    values, active = {}, False
    for line in path.read_text().splitlines():
        if line == "llm:":
            active = True
            continue
        if not active or not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith("  "):
            break
        match = re.fullmatch(r"  ([a-z_]+): ([^#]+)", line)
        if not match or match[1] in values:
            raise ValueError("unsupported LLM config syntax")
        key, value = match[1], match[2].strip()
        values[key] = (
            int(value) if value.isdecimal() else {"true": True, "false": False}.get(value, value)
        )
    for key in ("provider", "base_url", "model", "endpoint", "api_key_env"):
        if "MMAGENT_LLM_" + key.upper() in env:
            values[key] = env["MMAGENT_LLM_" + key.upper()]
    for key in ("timeout_seconds", "max_retries"):
        if "MMAGENT_LLM_" + key.upper() in env:
            try:
                values[key] = int(env["MMAGENT_LLM_" + key.upper()])
            except (TypeError, ValueError):
                raise ValueError("invalid nonsecret LLM configuration") from None
    try:
        return LLMConfig.model_validate(values)
    except ValueError:
        # ValidationError includes input values; never echo caller environment/config.
        raise ValidationError.from_exception_data(
            "LLMConfig",
            [
                {
                    "type": "value_error",
                    "loc": (),
                    "input": None,
                    "ctx": {"error": ValueError("invalid nonsecret LLM configuration")},
                }
            ],
        ) from None


class ExtractorSpec(Model):
    method: Literal["RULE", "LLM"]
    version: Version
    model: Term | None
    model_version: Term | None
    configuration_checksum: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class StructuredExtractor(Protocol):
    spec: ExtractorSpec
    live: bool

    def extract(self, request: dict, *, timeout: float, max_output_tokens: int) -> bytes: ...


class FakeStructuredExtractor:
    """Deterministic structured output fixture; no actual model quality assertion."""

    live = False

    def __init__(self, output: bytes):
        self.output = output
        self.calls = 0
        self.spec = ExtractorSpec(
            method="LLM",
            version="1.0.0",
            model="offline-structured-fixture",
            model_version="1.0.0",
            configuration_checksum=checksum(output),
        )

    def extract(self, request, *, timeout, max_output_tokens):
        self.calls += 1
        return self.output


class ConfiguredStructuredExtractor:
    live = True

    def __init__(self, config: LLMConfig):
        self.config = LLMConfig.model_validate(dict(vars(config)))
        if config.endpoint != "chat_completions":
            raise ValueError("legacy_extractor_requires_chat_completions")
        self.spec = ExtractorSpec(
            method="LLM",
            version="1.0.0",
            model=config.model,
            model_version="configured-provider-alias",
            configuration_checksum=digest(config),
        )

    def extract(self, request, *, timeout, max_output_tokens):
        try:
            result = subprocess.run(
                [sys.executable, "-m", "packages.knowledge_claims.worker"],
                input=canonical(
                    {
                        "config": self.config.model_dump(mode="json"),
                        "request": request,
                        "max_output_tokens": max_output_tokens,
                    }
                ),
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise Rejected("extractor_timeout") from None
        if result.returncode or len(result.stdout) > 65536:
            raise Rejected("extractor_transport")
        return result.stdout


class Budget:
    def __init__(self, policy: BatchPolicy, *, clock=time.monotonic):
        self.policy = BatchPolicy.model_validate(dict(vars(policy)))
        self.clock = clock
        self.deadline = clock() + policy.total_seconds
        self.requests = 0
        self.input_chars = 0

    def reserve(self, chars, timeout):
        left = self.deadline - self.clock()
        if (
            left <= 0
            or self.requests >= self.policy.max_requests
            or self.input_chars + chars > self.policy.max_input_chars
        ):
            raise Rejected("budget_exhausted")
        self.requests += 1
        self.input_chars += chars
        return min(timeout, left)


def extraction_request(evidence):
    return {
        "prompt": PROMPT.read_text(),
        "schema": StructuredOutput.model_json_schema(),
        "evidence": {
            "text": evidence.text,
            "span_start": evidence.chunk.location.span_start,
            "span_end": evidence.chunk.location.span_end,
            "coordinate_system": "canonical_unicode",
            "table": evidence.block.table.model_dump(mode="json") if evidence.block.table else None,
        },
    }


class RuleStructuredExtractor:
    """Registered local production adapter; source.synthetic remains independent."""

    live = False

    def __init__(self, request, *, producer):
        from .ingestion import parse_candidate_request, submission_for
        from .models import N124RuleExecution, RuleExecution
        from .n124_models import N124CandidateRequest

        self.request = parse_candidate_request(request)
        self.submission = submission_for(self.request, producer)
        n124 = isinstance(self.request, N124CandidateRequest)
        self.execution = N124RuleExecution() if n124 else RuleExecution()
        self.spec = ExtractorSpec(
            method="RULE",
            version="2.0.0" if n124 else "1.0.0",
            model=None,
            model_version=None,
            configuration_checksum=self.request.rule_config_checksum,
        )

    def validate(self, evidence):
        from .ingestion import validate_candidate

        return validate_candidate(self.request, evidence)

    def extract(self, request, *, timeout, max_output_tokens):
        from .ingestion import rule_output

        require_text = request["evidence"]
        if (
            request.get("candidate_request") != self.request.model_dump(mode="json")
            or require_text["text"] != self.request.candidate.quote
            or require_text["span_start"] != self.request.candidate.span.start
            or require_text["span_end"] != self.request.candidate.span.end
        ):
            raise Rejected("candidate_source_binding")
        return rule_output(self.request)
