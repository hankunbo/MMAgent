"""Source growth changes aliases, never old medical judgments or permissions."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from agents.expert_agent.reporting import read_research_report
from packages.evaluation_preparation.generic_aliases import alias_review
from packages.evidence_details.acceptance import ResearchAcceptance
from packages.evidence_details.acceptance_models import read_records
from packages.evidence_details.accepted_aggregation import (
    ACCEPTANCE_KEY,
    AcceptedOriginReader,
    AcceptedReviewSet,
    AggregationPermission,
    AggregationRequests,
    HistoricalRequests,
)
from packages.evidence_details.accepted_review_reuse import compatible_local_ids
from packages.evidence_details.accepted_source_aliases import project_aliases
from packages.evidence_details.review_batches import read_submission
from packages.grounding.models import digest, validate
from packages.planner_control.models import Control
from packages.research_runtime.agent_records import RuntimeAgentJournal
from tests.agent_provider.batch_support import generate
from tests.evaluation_preparation.test_directional_review import fixture
from tests.expert_agent.test_batch_assembly import LiveDocuments
from tests.expert_agent.test_legacy_accepted_aggregation import move_state


def prepared():
    f = fixture()
    original = f.owner.current()
    f.owner.state, old_map = alias_review(original)
    sub = generate(f)
    assert f.acceptance.submit(sub, key="original")["status"] == "submitted_not_accepted"
    assert f.acceptance.accept(key="original").status == "accepted"
    f.documents.run_id = f.binding.run_id
    control = validate(Control, dict(binding=f.binding, graph=f.binding.graph, phase="stopped"))
    f.documents.sessions = SimpleNamespace(read=lambda _: control)
    extra = original.material.sources[0].model_copy(
        update={
            "reference": original.material.sources[0].reference.model_copy(
                update={"code": "AAA:NEW"}
            ),
            "text": "Additional research source; not a new patient observation.",
        }
    )
    raw = original.model_copy(
        update={
            "material": original.material.model_copy(
                update={
                    "sources": (*original.material.sources, extra),
                }
            ),
        }
    )
    current, current_map = alias_review(raw)
    current = move_state(current, "RUN:ALIAS_AGGREGATED")
    reader = AcceptedOriginReader(
        documents=f.documents,
        requests=HistoricalRequests(f.budget.journal),
        state_reader=f.owner.current,
        acceptance_key="original",
        alias_manifests=lambda: (old_map, current_map),
    )
    source = AcceptedReviewSet([reader], f.config)
    docs = LiveDocuments()
    docs.run_id = current.material.scope.run_id
    docs.root = SimpleNamespace(checksum=current.material.scope.graph.checksum)
    docs.sessions = SimpleNamespace(
        read=lambda _: control.model_copy(
            update={
                "binding": f.binding.model_copy(
                    update={
                        "run_id": docs.run_id,
                        "case_code": current.material.scope.case_code,
                        "input_checksum": current.material.scope.input_checksum,
                        "visible_checksum": current.material.scope.visible_checksum,
                    }
                ),
            }
        )
    )
    owner = SimpleNamespace(current=lambda: current, accepted_review_set=source)
    requests = AggregationRequests(source, owner, RuntimeAgentJournal(docs))
    return SimpleNamespace(
        f=f,
        sub=sub,
        old=f.owner.current(),
        current=current,
        old_map=old_map,
        current_map=current_map,
        source=source,
        reader=reader,
        docs=docs,
        owner=owner,
        requests=requests,
    )


def test_alias_growth_original_owner_accept_score_readback_and_idempotency():
    a = prepared()
    before = digest(a.f.documents.saved), digest(a.f.budget.journal.read())
    with pytest.raises(ValueError, match="visible_context_changed"):
        compatible_local_ids(a.old, a.current, a.sub.candidate, include_unassessed=True)
    permission, sub = AggregationPermission.prepare(
        owner=a.owner,
        documents=a.docs,
        requests=a.requests,
        config=a.f.config,
    )
    assert sub.schema_version == "research-batched-submission/17.0.0"
    assert read_submission(sub) == sub
    service = ResearchAcceptance(
        owner=a.owner,
        documents=a.docs,
        requests=a.requests,
        live_permission=permission,
    )
    assert service.submit(sub, key=ACCEPTANCE_KEY)["status"] == "submitted_not_accepted"
    receipt = service.accept(key=ACCEPTANCE_KEY)
    assert receipt.status == "accepted"
    records = read_records(service.read(key=ACCEPTANCE_KEY)[0])
    assert records.schema_version == "case-observation-interpretation/2.19.0"
    old_items = {i.assessment_id: i for i in a.sub.candidate.items}
    for item in records.candidate.items:
        assert item.model_dump(exclude={"reference_citations"}) == old_items[
            item.assessment_id
        ].model_dump(exclude={"reference_citations"})
    report = read_research_report(service, key=ACCEPTANCE_KEY, parent=a.current)
    assert not report["scores"]["review_completion"]["pending_ids"]
    assert records.origins[0].alias_projection.transfers
    audit = list(a.docs.audit)
    assert service.accept(key=ACCEPTANCE_KEY) == receipt and a.docs.audit == audit
    assert before == (digest(a.f.documents.saved), digest(a.f.budget.journal.read()))
    assert a.requests.journal.read() == ()


@pytest.mark.parametrize(
    "fault",
    [
        "manifest",
        "duplicate_alias",
        "source_text",
        "source_identity",
        "definition",
        "observation",
        "config",
        "citation",
        "missing_owner",
        "duplicate_origin",
    ],
)
def test_alias_projection_keeps_all_semantic_and_owner_guards(fault):
    a = prepared()
    current, mapping = a.current, deepcopy(a.current_map)
    if fault == "manifest":
        mapping["checksum"] = digest("changed")
    elif fault == "duplicate_alias":
        mapping["bindings"][1] = mapping["bindings"][0]
    elif fault in {"source_text", "source_identity"}:
        s = current.material.sources[0]
        if fault == "source_text":
            s = s.model_copy(update={"text": s.text + " changed"})
        else:
            mapping["bindings"][1]["original"]["version"] = "wrong-version"
        current = current.model_copy(
            update={
                "material": current.material.model_copy(
                    update={
                        "sources": (s, *current.material.sources[1:]),
                    }
                )
            }
        )
    elif fault == "definition":
        p = current.policy.items[0]
        p = p.model_copy(
            update={
                "definition": p.definition.model_copy(
                    update={
                        "interpretation": "Changed medical interpretation",
                    }
                )
            }
        )
        current = current.model_copy(
            update={
                "policy": current.policy.model_copy(
                    update={
                        "items": (p, *current.policy.items[1:]),
                    }
                )
            }
        )
    elif fault == "observation":
        o = current.material.observations[0].model_copy(update={"value": "changed observation"})
        current = current.model_copy(
            update={
                "material": current.material.model_copy(
                    update={
                        "observations": (o, *current.material.observations[1:]),
                    }
                )
            }
        )
    elif fault == "config":
        current = current.model_copy(update={"config_checksum": digest("other-model")})
    elif fault == "citation":
        item = a.sub.candidate.items[0].model_copy(update={"reference_citations": ["SRC:UNKNOWN"]})
        candidate = a.sub.candidate.model_copy(update={"items": (item,)})
        with pytest.raises(ValueError, match="aggregation_alias_unknown_citation"):
            project_aliases(a.old, current, candidate, a.old_map, mapping)
        return
    elif fault == "missing_owner":
        a.reader.alias_manifests = None
    elif fault == "duplicate_origin":
        a.source.readers = (a.reader, a.reader)
    if fault != "missing_owner":
        a.reader.alias_manifests = lambda: (a.old_map, mapping)
    with pytest.raises(ValueError):
        a.source.current(current)
    assert not a.docs.saved and not a.requests.journal.read()


def test_alias_plan_is_explicit_and_old_plan_bytes_unchanged():
    from packages.multimodal_research.accepted_aggregation import read_plan

    old = dict(
        schema_version="accepted-review-aggregation-plan/1.3.0",
        facts="facts",
        origins=[dict(campaign="origin", local_finalization=None)],
        scope="all_current_obtained",
        accounting="fact_lineage_and_review_origins",
    )
    assert read_plan(old).model_dump(mode="json") == old
    new = dict(
        old,
        schema_version="accepted-review-aggregation-plan/1.6.0",
        reuse_kind="original_owner_exact_source_aliases",
    )
    assert read_plan(new).model_dump(mode="json") == new
