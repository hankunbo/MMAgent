"""Explicit test-only ABCD installer; uses the original ranking/query implementations."""

import json
from pathlib import Path

from packages.grounding.models import validate
from packages.memory_review.models import Query
from packages.memory_review.planner import MemoryRankedActions
from packages.planner_control.actions import RankedActions
from packages.retrieval.memory import MemoryRetrieval

CONFIG = Path(__file__).resolve().parents[2] / "configs/memory/10e-abcd-v1.json"
GROUPS = {
    "A": (),
    "B": ("success",),
    "C": ("success", "recovery"),
    "D": ("success", "counterfactual"),
}


def load_protocol(path=CONFIG):
    value = json.loads(Path(path).read_text())
    if (
        value.get("schema_version") != "memory-ablation-protocol/1.0.0"
        or value.get("protocol_id") != "10e-abcd-success-recovery-counterfactual-v1"
        or value.get("scope") != "synthetic_offline_engineering"
        or value.get("groups")
        != {
            k: {"install_memory_ranked_actions": bool(groups), "query_groups": list(groups)}
            for k, groups in GROUPS.items()
        }
    ):
        raise ValueError("10e_protocol_mismatch")
    return value


class GroupQueryReader:
    def __init__(self, owner, groups):
        self.owner, self.groups = owner, tuple(groups)

    def current(self, run_id):
        current = validate(Query, self.owner.current(run_id))
        return validate(Query, current.model_copy(update={"groups": self.groups}))

    def check_versions(self, query):
        return self.owner.check_versions(query)


def install(group, source, *, query_reader, repository, documents, policy=None, agent=None):
    """Contract harness only. This does not prepare sources or qualify an experiment."""
    setting = load_protocol()["groups"][group]
    if not setting["install_memory_ranked_actions"]:
        return RankedActions(source, policy=policy, agent=agent)
    retrieval = MemoryRetrieval(
        repository=repository,
        query_reader=GroupQueryReader(query_reader, setting["query_groups"]),
    )
    return MemoryRankedActions(
        source, retrieval=retrieval, documents=documents, policy=policy, agent=agent
    )
