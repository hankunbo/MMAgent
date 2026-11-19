"""Append-only correction of a never-dispatched capability Review origin.

The original Runtime claim and protocol remain immutable. This cannot renew
time, budget, patient inputs, acquisition targets or an already started run.
"""

import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from apps.vectorengine_single_case import write_once
from db.runtime.repository.case_pipeline import identity
from packages.grounding.models import digest

from . import capability_window as window

VERSION = "capability-review-origin-amendment/1.0.0"
PATH = "capability-review-origin-amendment.json"
ALLOWED = {
    "packages/multimodal_research/capability_window.py",
    "packages/multimodal_research/capability_review_origin.py",
    "packages/multimodal_research/campaign_renewal.py",
    "apps/multimodal_case.py",
}


def key(old):
    return "capability:review-origin-amendment:" + digest(old)[7:]


def revised(old, found, implementation):
    if old.get("schema_version") != window.VERSION:
        raise ValueError("capability_review_amendment_version")
    inv = deepcopy(old["capability_transition"])
    before = set(inv["confirmed_review"]["assessment_ids"])
    after = set(found["confirmed_review"]["assessment_ids"])
    inv["confirmed_review"] = found["confirmed_review"]
    if inv != found:
        changed_fields = sorted(k for k in inv.keys() | found.keys() if inv.get(k) != found.get(k))
        raise ValueError(
            "capability_review_amendment_inventory_changed:" + ",".join(changed_fields)
        )
    if not before < after or not after <= set(old["review_target_ids"]):
        raise ValueError("capability_review_amendment_only_confirmed_extension")
    changed = {
        k
        for k in old["implementation"].keys() | implementation.keys()
        if old["implementation"].get(k) != implementation.get(k)
    }
    if not changed or not changed <= ALLOWED:
        raise ValueError("capability_review_amendment_implementation_scope")
    plan = deepcopy(old)
    plan.update(
        capability_transition=found,
        reuse_assessment_ids=found["confirmed_review"]["assessment_ids"],
        implementation=implementation,
    )
    return plan


def documents(parent, old):
    docs = parent.documents(old["tip"]["run_id"])
    if (
        str(parent.output.absolute()) != old["facts"]
        or digest(json.loads((parent.output / "result.json").read_bytes()))
        != old["original_fact_result"]
        or docs.read("authorized-campaign-renewal") != (window.claim(old, old["output"]),)
    ):
        raise ValueError("capability_review_amendment_original_claim")
    return docs


def verify_unused(parent, old):
    out = Path(old["output"])
    if any(p.is_symlink() for p in (out, *out.parents)):
        raise ValueError("capability_review_amendment_symlink")
    if any(
        (out / name).exists()
        for name in ("result.json", "stage-failure.json", "stage-closure-error.json", "cancel")
    ) or list(out.glob("**/request.json")):
        raise ValueError("capability_review_amendment_started_or_cancelled")
    targets = (None, *old["acquisition_targets"], *old["input_scope"]["targets"])
    for target in targets:
        request_key = "mm-live:" + digest((out.name, target or "review"))[7:55]
        rid = identity("RUN", "task-research:" + request_key)
        if (
            parent.contexts.read_research_run(rid, missing_ok=True) is not None
            or parent.sessions.read(rid) is not None
        ):
            raise ValueError("capability_review_amendment_phase_registered")


def verify_proof(path, implementation):
    proof = json.loads(Path(path).read_bytes())
    if (
        proof.get("schema_version") != "capability-review-origin-tests/1.0.0"
        or proof.get("implementation") != implementation
        or not {"cpu", "pg", "lint", "format"} <= proof.get("checks", {}).keys()
    ):
        raise ValueError("capability_review_amendment_test_proof")
    from packages.object_store import checksum

    for check in proof["checks"].values():
        report = json.loads(Path(check["path"]).read_bytes())
        if (
            digest(report) != check["checksum"]
            or report.get("direct_exit") != 0
            or checksum(Path(check["log"]).read_bytes()) != report["log_checksum"]
        ):
            raise ValueError("capability_review_amendment_test_result")
    return proof


def resolve(parent, old):
    """Recheck original Runtime receipt, immutable local mirror and tested code."""
    docs = documents(parent, old)
    saved = docs.read(key(old))
    path = Path(old["output"]) / PATH
    if not saved:
        if path.exists():
            raise ValueError("capability_review_amendment_uncommitted")
        return old
    if len(saved) != 1 or not path.exists():
        raise ValueError("capability_review_amendment_local_recovery_required")
    value = json.loads(path.read_bytes())
    if (
        saved != (value,)
        or value.get("schema_version") != VERSION
        or value.get("original_protocol") != digest(old)
        or value.get("output") != old["output"]
        or digest(verify_proof(value["test_proof"], value["plan"]["implementation"]))
        != value["test_proof_checksum"]
        or revised(old, value["plan"]["capability_transition"], value["plan"]["implementation"])
        != value["plan"]
    ):
        raise ValueError("capability_review_amendment_binding")
    return value["plan"]


def prepare(*, output, test_proof, authorization):
    from packages.dataset_adapters.current_read_scope import _projection_visible_reads
    from packages.evidence_details.accepted_read_scope import historical_fact_read_scope

    from .campaign_renewal import completed_parent
    from .execution_lease import session
    from .implementation import freeze

    out = Path(output).absolute()
    old = json.loads((out / "protocol.json").read_bytes())
    if str(out) != old["output"] or not authorization.strip() or len(authorization) > 200:
        raise ValueError("capability_review_amendment_identity")
    with historical_fact_read_scope(), _projection_visible_reads():
        parent, _ = completed_parent(old["facts"])
        lease = SimpleNamespace(output=out, target=parent.target, base=parent.base)
        with session(lease):
            docs = documents(parent, old)
            saved = docs.read(key(old))
            if saved:
                if len(saved) != 1:
                    raise ValueError("capability_review_amendment_binding")
                write_once(out / PATH, saved[0])
                plan = resolve(parent, old)
                if plan["implementation"] != freeze():
                    raise ValueError("capability_review_amendment_code_changed")
                return dict(status="already_recorded", api_requests=0, database_writes=0)
            verify_unused(parent, old)
            implementation = freeze()
            proof = verify_proof(test_proof, implementation)
            _, found, _ = window.inventory(parent, out, frozen_methods=old["procedure_reuse"])
            plan = revised(old, found, implementation)
            # Inventory uses the original qualified reader, including exact old
            # answer preservation. Recheck unstarted/deadline after those reads.
            verify_unused(parent, old)
            if datetime.now(UTC) >= datetime.fromisoformat(old["deadline"]):
                raise ValueError("capability_review_amendment_expired")
            value = dict(
                schema_version=VERSION,
                original_protocol=digest(old),
                output=str(out),
                authorization=authorization,
                test_proof=str(Path(test_proof).absolute()),
                test_proof_checksum=digest(proof),
                plan=plan,
            )
            with docs.session():
                docs.write(key(old), (value,))
            write_once(out / PATH, value)
    return dict(
        status="recorded",
        retained_assessment_ids=plan["reuse_assessment_ids"],
        deadline=plan["deadline"],
        previous_requests=sum(a["requests"] for a in plan["previous_attempts"]),
        max_new_requests=plan["max_new_requests"],
        api_requests=0,
        database_writes=1,
    )
