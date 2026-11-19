"""Frozen experiment inputs select identities, never confer current authorization."""

import hashlib
import json
from pathlib import Path

from packages.grounding.models import digest


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return {"path": Path(path).name, "sha256": sha(path)}


def reference(base, ref):
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
        raise ValueError("10e_reference_shape")
    path = base / ref["path"]
    if path.resolve().parent != base.resolve() or sha(path) != ref["sha256"]:
        raise ValueError("10e_reference_digest")
    return json.loads(path.read_text())


def independent(supports, queries):
    if not supports or not queries:
        raise ValueError("10e_split_empty")
    for query in queries:
        if any(
            query["case_code"] == s["case_code"] or query["patient_hash"] == s["patient_hash"]
            for s in supports
        ):
            raise ValueError("10e_support_query_overlap")


def validate_actual(value, base):
    if (
        value.get("execution_status") != "frozen_before_comparison"
        or value.get("live_domain_acceptance") != "live_domain_acceptance_not_enabled"
        or value.get("real_mem001_approved") is not False
        or value.get("support_query_case_overlap_allowed") is not False
    ):
        raise ValueError("10e_actual_scope")
    pool = reference(base, value.get("memory_pool_freeze"))
    profile = reference(base, value.get("task_profile_ref"))
    if not value.get("task_profile") or profile["task_profile"] != value["task_profile"]:
        raise ValueError("10e_profile_binding")
    if set(pool["entries"]) != {"success", "recovery", "counterfactual"}:
        raise ValueError("10e_pool_groups")
    for group, entries in pool["entries"].items():
        if not entries:
            raise ValueError("10e_pool_empty:" + group)
        for entry in entries:
            if entry["group"] != group:
                raise ValueError("10e_pool_group_binding")
            independent(entry["supports"], pool["queries"])
    return pool


class FrozenPool:
    """Test installer restriction atop original current repository and Runtime reads."""

    def __init__(self, path, checksum, repository):
        from tests.research_task_completion.protocol import load_protocol

        self.path, self.checksum, self.repository = Path(path), checksum, repository
        self.contexts = repository.contexts
        self.protocol = load_protocol(path, expected_sha256=checksum)
        self.pool = validate_actual(self.protocol, self.path.parent)
        self.entries = {
            e["publication"]: e for entries in self.pool["entries"].values() for e in entries
        }
        if len(self.entries) != sum(map(len, self.pool["entries"].values())):
            raise ValueError("10e_duplicate_pool_identity")

    def check_query(self, query):
        from tests.research_task_completion.protocol import load_protocol

        load_protocol(self.path, expected_sha256=self.checksum)
        if query.task_profile != self.protocol["task_profile"]:
            raise ValueError("10e_query_profile")
        scope = self.contexts.read_research_run(query.run_id).request.scope
        identity = dict(
            run_id=query.run_id, case_code=scope.case_code, patient_hash=scope.patient_hash
        )
        if identity not in self.pool["queries"] or query.case_code != scope.case_code:
            raise ValueError("10e_query_identity")
        for entry in self.entries.values():
            supports = []
            for expected in entry["supports"]:
                current = self.contexts.read_research_run(expected["run_id"]).request.scope
                actual = dict(
                    run_id=expected["run_id"],
                    case_code=current.case_code,
                    patient_hash=current.patient_hash,
                )
                if actual != expected:
                    raise ValueError("10e_support_identity_drift")
                supports.append(actual)
            independent(supports, [identity])

    def publication_keys(self):
        return tuple(sorted(self.entries))

    def read_publication(self, key):
        entry = self.entries[key]
        publication = self.repository.read_publication(key)
        record = self.repository.read(publication.binding.target).record
        if (
            digest(publication) != entry["publication_checksum"]
            or publication.binding.group != entry["group"]
            or publication.schema_version != entry["qualification_version"]
            or record.key != entry["target"]
            or digest(record) != entry["checksum"]
            or record.schema_version != entry["content_version"]
            or {s.run_id for s in record.sources} != {s["run_id"] for s in entry["supports"]}
        ):
            raise ValueError("10e_pool_current_binding")
        return publication

    def read(self, *args, **kwargs):
        return self.repository.read(*args, **kwargs)
