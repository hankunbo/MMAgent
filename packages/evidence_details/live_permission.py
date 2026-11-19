"""Explicit local research authority. Never inferred from a key or model output."""

from datetime import datetime
from typing import Literal

from pydantic import Field

from packages.grounding.models import Checksum, Code, Value, digest, validate
from packages.llm_provider.config import read_provider_config


class ResearchLiveGrant(Value):
    schema_version: Literal["research-live-grant/1.0.0"] = "research-live-grant/1.0.0"
    enabled: Literal[True]
    case_code: Code
    run_id: Code
    input_checksum: Checksum
    visible_checksum: Checksum
    policy_checksum: Checksum
    config_checksum: Checksum
    root_checksum: Checksum
    purpose: Literal["bounded_single_case_research"]
    deadline: datetime
    max_requests: Literal[40] = 40
    max_steps: Literal[20] = 20
    max_no_progress: Literal[2] = 2
    max_searches: Literal[8] = 8
    seconds: Literal[2700] = 2700
    formal_graph: Literal[False] = False
    publication: Literal[False] = False
    roles: tuple[Literal["source", "vision", "review"], ...] = Field(min_length=1, max_length=3)


def validate_live_config(config):
    config = read_provider_config(config)
    t = config.transport
    if (
        config.mode != "live"
        or (t.provider, t.base_url, t.endpoint, t.model, t.api_key_env)
        not in {
            (
                "openai_compatible",
                "https://api.vectorengine.cn/v1",
                "chat_completions",
                "gpt-5.6-sol",
                "MMAGENT_VECTORENGINE_API_KEY",
            ),
            (
                "openai_compatible",
                "https://api.relayrouter.ai/v1",
                "chat_completions",
                "gpt-6-sol",
                "MMAGENT_VECTORENGINE_API_KEY",
            ),
            (
                "openai_compatible",
                "https://api.relayrouter.ai/v1",
                "chat_completions",
                "gpt-4o-2024-08-06",
                "MMAGENT_VECTORENGINE_API_KEY",
            ),
        }
        or t.max_retries not in {0, 1}
        or config.max_requests != 40
        or config.max_concurrency != 1
        or not config.image_capable
    ):
        raise ValueError("bounded_live_configuration_required")
    return config


class ResearchLivePermission:
    """Installed only by the local composition root, persisted in the original Runtime.

    Grant JSON is not an HTTP credential. Every use rechecks its immutable Runtime
    document, the exact run/input/policy/config, current owner, and absolute deadline.
    """

    def __init__(self, *, grant, config, documents, budget):
        self.grant = validate(ResearchLiveGrant, grant)
        self.config = validate_live_config(config)
        self.documents, self.budget = documents, budget
        g = self.grant
        if (
            not budget.journal.durable
            or g.config_checksum != digest(config)
            or g.run_id != documents.run_id
            or g.root_checksum != documents.root.checksum
            or g.run_id != budget.run_id
            or g.deadline != budget.deadline
            or budget.max_requests != g.max_requests
            or len(set(g.roles)) != len(g.roles)
            or g.deadline.tzinfo is None
        ):
            raise ValueError("research_live_grant_binding")
        with documents.session():
            documents.write("research-live-grant", (g,))

    def check(self, state, *, role=None, record=None):
        g, s = self.grant, state.material.scope
        if self.documents.read("research-live-grant") != (g.model_dump(mode="json"),):
            raise ValueError("research_live_grant_changed")
        if self.budget.clock() >= g.deadline:
            raise ValueError("deadline")
        if (
            (s.case_code, s.run_id, s.input_checksum, s.visible_checksum, s.policy_checksum)
            != (g.case_code, g.run_id, g.input_checksum, g.visible_checksum, g.policy_checksum)
            or state.config_checksum != g.config_checksum
            or s.graph.kind != "research_artifact"
            or any(x.purpose != "research_material" for x in state.material.sources)
            or (role is not None and role not in g.roles)
        ):
            raise ValueError("research_live_scope_forbidden")
        self.documents.validate_scope(s)
        if record is not None and (
            record.simulated
            or record.api_requests != 1
            or record.error is not None
            or record.config_checksum != g.config_checksum
            or record.scope != s
            or (record.provider, record.model, record.endpoint)
            != (
                self.config.transport.provider,
                self.config.transport.model,
                self.config.transport.endpoint,
            )
        ):
            raise ValueError("research_live_request_required")
        if record is not None:
            self.budget.check_control()


def require_live_permission(permission, state, *, role=None, record=None):
    if not isinstance(permission, ResearchLivePermission):
        raise ValueError("live_domain_acceptance_not_enabled")
    permission.check(state, role=role, record=record)
