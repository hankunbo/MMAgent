"""Keep a completed non-applied definition merge as history, not new authority."""

import json
from pathlib import Path

from packages.evidence_reuse.definition_acceptance import ResearchDefinitionOwner
from packages.grounding.models import digest

POLICY = "retain-resolved-definition-merges/1.0.0"


def policy_reference(reference, *, routed):
    value = reference.get("resolved_decisions_policy")
    if "resolved_decisions_policy" in reference and (not routed or value != POLICY):
        raise ValueError("definition_resolved_merge_policy")
    return {} if value is None else {"resolved_decisions_policy": POLICY}


def resolved_merge(decision, proposal, *, documents, parent, protocol, spec, index):
    """Verify original owner receipts without consuming their unused new relation.

    Current baseline source qualification remains the new campaign's normal
    responsibility. Historical equality is not medical or publication approval.
    """
    from .definition_expansion import read_expansion_policy
    from .gap_expansion import classify

    candidate, binding = decision["candidate"], decision["binding"]
    profile = json.loads(Path(spec["active_acquisition"]["path"]).read_bytes())
    if (
        digest(profile) != spec["active_acquisition"]["checksum"]
        or binding["policy_checksum"]
        != digest(read_expansion_policy(profile["definition_expansion"]))
        or decision["schema_version"] != "definition-gap-attempt/1.0.0"
        or decision["attempt"] != index
        or binding["campaign_id"] != parent.name
        or binding["input_checksum"] != spec["input_checksum"]
        or binding["target"] != spec["intent"]["target_code"]
        or binding["protocol_checksum"] != digest(protocol)
        or digest(candidate["rows"]) != binding["parent_requirements_checksum"]
        or candidate["added"]
        or candidate["replaced"]
        or not candidate["merged"]
        or not proposal.definitions
        or decision["decision"] != dict(classify(candidate), action="try_distinct_public_direction")
    ):
        raise ValueError("definition_resolved_merge_not_baseline")
    drafts = {digest(d)[7:47]: d for d in proposal.definitions}
    accepted, merged = candidate["accepted"], candidate["merged"]
    if len(drafts) != len(proposal.definitions) or set(drafts) != set(accepted):
        raise ValueError("definition_resolved_merge_unaccounted_draft")
    ids = {r["id"] for r in candidate["rows"]}
    if {m["definition"] for m in merged} != set(accepted) or any(
        m["existing"] not in ids
        or m["disposition"] != "existing_definition_kept_new_relation_not_applied"
        or m["reason"] != "existing_examination_requires_no_duplicate_vote"
        for m in merged
    ):
        raise ValueError("definition_resolved_merge_disposition")
    owner = ResearchDefinitionOwner(
        documents=documents,
        sources=None,
        guard=None,
        campaign_id=binding["campaign_id"],
        input_checksum=binding["input_checksum"],
        target=binding["target"],
    )
    receipts = []
    for key, value in accepted.items():
        if owner.read(key, historical=True) != value or digest(value["draft"]) != digest(
            drafts[key]
        ):
            raise ValueError("definition_resolved_merge_receipt")
        receipts.append(dict(key=key, checksum=digest(value)))
    return dict(
        schema_version=POLICY,
        original_run_id=documents.run_id,
        decision_key=f"definition:gap:{index}:decision",
        decision_checksum=digest(decision),
        baseline_requirements_checksum=binding["parent_requirements_checksum"],
        original_acceptances=receipts,
        original_merged=merged,
        historical_only=True,
        new_relation_applied=False,
        current_definition_qualification=False,
        new_source_requests=0,
    )
