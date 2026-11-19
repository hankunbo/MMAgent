"""Explicit test-only ABCD installer; uses the original ranking/query implementations."""

import hashlib
import json
from pathlib import Path

from packages.grounding.models import validate
from packages.memory_review.planner import MemoryRankedActions
from packages.memory_review.task_models import TaskQuery
from packages.planner_control.actions import RankedActions
from packages.retrieval.memory import MemoryRetrieval

CONFIG = Path(__file__).resolve().parents[2] / "configs/memory/10e-abcd-task-v1.json"
GROUPS = {
    "A": (),
    "B": ("success",),
    "C": ("success", "recovery"),
    "D": ("success", "counterfactual"),
}


def load_protocol(path=CONFIG, *, expected_sha256=None):
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    actual = expected_sha256 is not None
    if actual and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("10e_protocol_digest")
    if (
        value.get("schema_version") != "memory-ablation-protocol/1.1.0"
        or value.get("protocol_id")
        != ("10e-abcd-actual-v1" if actual else "10e-abcd-task-success-v1")
        or value.get("query_schema") != "memory-query/1.1.0"
        or value.get("success_semantics") != "research-task-completion/1.0.0"
        or value.get("scope") != "synthetic_offline_engineering"
        or value.get("groups")
        != {
            k: {"install_memory_ranked_actions": bool(groups), "query_groups": list(groups)}
            for k, groups in GROUPS.items()
        }
    ):
        raise ValueError("10e_protocol_mismatch")
    if actual:
        from tests.memory_ablation.freeze import validate_actual

        validate_actual(value, Path(path).parent)
    return value


class GroupQueryReader:
    def __init__(self, owner, groups, *, frozen=None):
        self.owner, self.groups, self.frozen = owner, tuple(groups), frozen

    def current(self, run_id):
        current = validate(TaskQuery, self.owner.current(run_id))
        if self.frozen:
            self.frozen.check_query(current)
        return validate(TaskQuery, current.model_copy(update={"groups": self.groups}))

    def check_versions(self, query):
        if self.frozen:
            self.frozen.check_query(query)
        return self.owner.check_versions(query)


def install(
    group,
    source,
    *,
    query_reader,
    repository,
    documents,
    policy=None,
    agent=None,
    protocol_path=CONFIG,
    protocol_sha256=None,
):
    """Default remains the historical template; actual runs require explicit frozen bytes."""
    protocol = load_protocol(protocol_path, expected_sha256=protocol_sha256)
    setting = protocol["groups"][group]
    frozen = None
    if protocol_sha256 is not None:
        from tests.memory_ablation.freeze import FrozenPool

        frozen = FrozenPool(protocol_path, protocol_sha256, repository)
        repository = frozen
    if not setting["install_memory_ranked_actions"]:
        ranking = RankedActions(source, policy=policy, agent=agent)
    else:
        retrieval = MemoryRetrieval(
            repository=repository,
            query_reader=GroupQueryReader(query_reader, setting["query_groups"], frozen=frozen),
            optional=protocol_sha256 is None,
        )
        ranking = MemoryRankedActions(
            source, retrieval=retrieval, documents=documents, policy=policy, agent=agent
        )
    ranking.experiment_binding = {
        "protocol_sha256": protocol_sha256,
        "task_profile": protocol["task_profile"],
        "memory_pool_freeze": protocol["memory_pool_freeze"],
        "group": group,
    }
    return ranking
