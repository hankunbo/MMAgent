"""Exact current sources retain prior names; answers and historical bytes stay fixed."""

from unittest.mock import patch

import pytest

from packages.evaluation_preparation import generic_aliases
from packages.evidence_acquisition.source_aliases import alias_sources as historical_aliases
from packages.grounding.models import digest
from packages.llm_provider.models import Source


def module(name):
    from packages.evaluation_preparation import generic_aliases
    from packages.evidence_acquisition import source_aliases

    return generic_aliases if name == "generic_aliases" else source_aliases


def source(code, text):
    return Source(
        reference=dict(code=code, role="source", version="1.0.0", checksum=digest(text)),
        locator=code,
        purpose="research_material",
        state="research_only",
        text=text,
    )


def test_inserted_earlier_source_keeps_previous_exact_codes():
    before = (source("SOURCE:B", "B exact passage"), source("SOURCE:C", "C exact passage"))
    previous, _ = historical_aliases(before)
    current = (source("SOURCE:A", "A new passage"), *before)
    old_hash = digest(previous)
    aliases, proof = module("source_aliases").alias_sources(current, previous=previous)
    assert aliases[1:] == previous and aliases[0].reference.code == "SRC:03"
    assert proof["schema_version"] == "research-source-aliases/1.1.0"
    assert proof["prior_sources_checksum"] == old_hash == digest(previous)
    assert module("source_aliases").alias_sources(current) == historical_aliases(current)


@pytest.mark.parametrize("fault", ["missing", "changed", "forged_alias", "duplicate_alias"])
def test_original_source_or_alias_change_rejects(fault):
    before = (source("SOURCE:B", "B exact passage"), source("SOURCE:C", "C exact passage"))
    previous, _ = historical_aliases(before)
    current = before
    if fault == "missing":
        current = before[:1]
    elif fault == "changed":
        current = (before[0].model_copy(update={"text": "Different passage"}), before[1])
    elif fault == "forged_alias":
        previous = (
            previous[0].model_copy(update={"text": "Different historical passage"}),
            previous[1],
        )
    else:
        previous = (*previous, previous[0])
    with pytest.raises(ValueError, match="source_alias_prior_"):
        module("source_aliases").alias_sources(current, previous=previous)


def test_original_domain_reuse_keeps_exact_policy_under_expansion():
    from packages.evidence_details.accepted_review_reuse import compatible_local_ids
    from tests.agent_provider.test_accepted_review_reuse import accepted_fixture, state_from

    f = accepted_fixture()
    old = f.old.owner.current()
    added = source("SOURCE:000_NEW", "Additional source with separate provenance.")
    raw = old.model_dump(mode="json")
    raw["material"]["sources"].append(added.model_dump(mode="json"))
    expanded = state_from(raw)
    original, _ = generic_aliases.alias_review(old)
    candidate = module("generic_aliases")
    with patch.object(candidate, "alias_sources", module("source_aliases").alias_sources):
        aligned, _ = candidate.alias_review(expanded, previous=original.material.sources)
    # This checks actual original policy/visible-context reuse, not alias helpers alone.
    origin = f.owner.confirmed_review_reader.current(f.owner.current())
    ids = compatible_local_ids(original, aligned, origin.submission.candidate)
    assert ids
    assert original.policy == aligned.policy
