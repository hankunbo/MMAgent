"""New aliases reduce copying risk without changing observations or accepting typos."""

import json

import pytest

from packages.evaluation_preparation.comprehensive_campaign import ComprehensiveCampaign
from packages.evaluation_preparation.generic_aliases import alias_review
from packages.evaluation_preparation.generic_campaign import GenericCampaign
from packages.evidence_details.review_tasks import project_review_tasks
from packages.grounding.models import digest
from packages.llm_provider.config import prompt, read_provider_config
from tests.agent_provider.batch_support import generate
from tests.evaluation_preparation.test_dynamic_evidence import dynamic_fixture


def test_aliases_bind_requirements_items_and_current_sources_then_original_acceptance():
    f = dynamic_fixture()
    old = f.owner.current()
    new, manifest = alias_review(old)
    assert len(manifest["bindings"]) == len(old.material.sources)
    assert new.material.observations == old.material.observations
    assert new.policy.groups == old.policy.groups
    assert new.resources == old.resources
    assert new.policy.aggregation_checksum == old.policy.aggregation_checksum
    by_code = {s.reference.code: s for s in old.material.sources}
    for binding in manifest["bindings"]:
        source = by_code[binding["original"]["code"]]
        assert binding["source_checksum"] == digest(source)
        assert binding["alias"]["code"].startswith("SRC:")
    definitions = {r.id: r.definition for r in new.policy.requirements}
    for p in new.policy.items:
        assert p.definition == definitions[p.id]
        assert all(c.startswith("SRC:") for c in p.source_codes)
    project_review_tasks(new)
    f.owner.state = new
    proposal = generate(f)
    f.acceptance.submit(proposal, key="short-sources")
    assert f.acceptance.accept(key="short-sources").status == "accepted"
    # Different current source text yields a different projection, not cached authorization.
    source = old.material.sources[0]
    changed = old.model_copy(
        update={
            "material": old.material.model_copy(
                update={
                    "sources": (
                        source.model_copy(update={"text": source.text + " changed"}),
                        *old.material.sources[1:],
                    )
                }
            )
        }
    )
    assert alias_review(changed)[1] != manifest


@pytest.mark.parametrize("model", [None, dict(source=dict(original="confirmed"))])
def test_confirmed_unreviewed_model_has_process_row_but_absent_model_is_not_slot(
    monkeypatch, model
):
    monkeypatch.setattr(ComprehensiveCampaign, "process_evaluations", lambda _: [])
    c = object.__new__(GenericCampaign)
    c.read_model = lambda: model
    rows = c.process_evaluations()
    if model is None:
        assert rows == []
    else:
        assert rows[0]["availability_score"] == 1
        assert rows[0]["disease_direction"] is None
        assert rows[0]["interpretation_status"] == "pending_review"


def test_generic_prompt_is_versioned_and_missing_judgment_remains_null():
    from pathlib import Path

    from packages.evaluation_preparation.generic_input import PROVIDER
    from packages.llm_provider.review_citations import PROMPT_VERSIONS

    config = read_provider_config(json.loads(Path(PROVIDER).read_bytes()))
    setting, content = prompt(config, "review")
    assert setting.prompt_version == "review-batch-1.7.0"
    assert setting.prompt_version in PROMPT_VERSIONS
    assert "MUST be null" in content and "exact short SRC code" in content
