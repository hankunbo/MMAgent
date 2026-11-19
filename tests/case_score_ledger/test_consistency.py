"""Standard regressions for the independent H review; all inputs are synthetic."""

import json
from dataclasses import replace

import pytest

from packages.case_score_ledger.models import Component, ScoreEntry, ScoreEnvelope, ValidationFact
from packages.case_score_ledger.service import CaseScoreService, definition
from packages.grounding.models import canonical, digest, seal, validate
from tests.case_score_ledger.test_integration import counts, setup
from tests.case_score_ledger.test_models import source


def accepted_envelope():
    """Small model fixture; the PG test below uses an actual newly saved envelope."""
    base = source()
    graph = dict(graph_code="GRAPH:SYNTHETIC", graph_version=3, graph_checksum=digest("graph3"))
    facts = tuple(
        validate(
            ValidationFact,
            dict(
                validation_code=f"VALIDATION:SYNTHETIC_{i}",
                validation_checksum=digest(f"validation{i}"),
                binding_checksum=digest(f"binding{i}"),
                target_node_code=f"NODE:SYNTHETIC_{i}",
                clinical_effect="SUPPORTS",
                source_snapshot={
                    **graph,
                    "graph_version": i,
                    "graph_checksum": digest(f"graph{i}"),
                },
                accepted_graph={
                    **graph,
                    "graph_version": i + 1,
                    "graph_checksum": digest(f"graph{i + 1}"),
                },
                audit_event_id=f"MESSAGE:SYNTHETIC_{i}",
                audit_event_sequence=i,
                rule_refs=(),
                evidence_bindings=(f"ASSET:SYNTHETIC_{i}",),
            ),
        )
        for i in (1, 2)
    )
    graph_source = base.model_copy(
        update=dict(
            qualification="accepted_graph",
            graph=validate(type(facts[0].accepted_graph), graph),
            accepted_event_ids=tuple(v.audit_event_id for v in facts),
            validation_checksums=tuple(v.validation_checksum for v in facts),
        )
    )
    entry = ScoreEntry(
        definition=definition("support"),
        target="GOAL:SYNTHETIC",
        status="available",
        value=Component(
            contributing_nodes=(),
            rule_refs=(),
            required_count=2,
            resolved_count=2,
            blocked_count=0,
            explanations=(),
        ),
        missing_reason=None,
        source=graph_source,
    )
    return seal(
        ScoreEnvelope,
        scope=base.scope,
        synthetic=True,
        policy_checksum=base.policy_checksum,
        entries=(entry,),
        technical=(),
        graph=graph,
        graph_status="accepted",
        original_report=None,
        accepted_validations=facts,
        run_status="COMPLETED",
    )


MUTATIONS = {
    "graph_status": "score_graph_qualification",
    "source_policy": "score_source_policy_mismatch",
    "no_validations": "score_graph_qualification",
    "wrong_graph": "score_validation_graph_mismatch",
    "wrong_snapshot": "score_validation_graph_mismatch",
    "wrong_entry_graph": "score_validation_attribution_mismatch",
    "unpaired": "score_validation_attribution_mismatch",
    "duplicate_fact": "score_validation_identity",
}


def mutate(envelope, change):
    data = envelope.model_dump(mode="json")
    graph_entry = next(
        e for e in data["entries"] if e["source"]["qualification"] == "accepted_graph"
    )
    if change == "graph_status":
        data["graph_status"] = "unavailable"
    elif change == "source_policy":
        data["entries"][0]["source"]["policy_checksum"] = digest("unrelated score policy")
    elif change == "no_validations":
        data["accepted_validations"] = []
        data["entries"] = [
            e for e in data["entries"] if e["source"]["qualification"] != "accepted_graph"
        ]
    elif change == "wrong_graph":
        data["accepted_validations"][0]["accepted_graph"]["graph_code"] = "GRAPH:OTHER"
    elif change == "wrong_snapshot":
        data["accepted_validations"][-1]["accepted_graph"].update(
            graph_version=data["graph"]["graph_version"], graph_checksum=digest("wrong snapshot")
        )
    elif change == "wrong_entry_graph":
        graph_entry["source"]["graph"]["graph_checksum"] = digest("wrong source graph")
    elif change == "unpaired":
        graph_entry["source"]["accepted_event_ids"] = ["MESSAGE:UNRELATED"]
    elif change == "duplicate_fact":
        data["accepted_validations"].append(data["accepted_validations"][0])
    data.pop("checksum")
    return {**data, "checksum": digest(data)}


@pytest.mark.parametrize("change", MUTATIONS)
def test_public_envelope_rejects_resealed_inconsistency(change):
    original = accepted_envelope()
    data = mutate(original, change)
    for candidate in (
        data,
        original.model_copy(update=data),
        ScoreEnvelope.model_construct(**data),
    ):
        with pytest.raises(ValueError, match=MUTATIONS[change]):
            validate(ScoreEnvelope, candidate)
    with pytest.raises(ValueError, match=MUTATIONS[change]):
        ScoreEnvelope.model_validate_json(json.dumps(data))


def test_event_validation_pairs_cannot_borrow_each_others_authority():
    original = accepted_envelope()
    data = original.model_dump(mode="json")
    data["entries"][0]["source"]["validation_checksums"].reverse()
    with pytest.raises(ValueError, match="score_validation_attribution_mismatch"):
        seal(ScoreEnvelope, **data)


def test_complete_historical_acceptances_and_local_facts_remain_legal():
    original = accepted_envelope()
    assert ScoreEnvelope.model_validate_json(original.model_dump_json()) == original
    # An older accepted snapshot remains legal; a partial fact set is not whole-graph scoring.
    local = seal(
        ScoreEnvelope,
        **{**original.model_dump(mode="json"), "graph_status": "unavailable", "entries": []},
    )
    assert local.accepted_validations and local.graph_status == "unavailable"
    research = seal(
        ScoreEnvelope,
        **{**local.model_dump(mode="json"), "graph": None, "accepted_validations": []},
    )
    assert research.graph is None


def test_pg_review_refusals_before_stage_and_unchanged_ledger(assembled_pipeline, monkeypatch):
    p = assembled_pipeline
    _, state, service = setup(p)
    envelope = service.save(state.run_id, "consistency-control")
    assert (
        service.save(state.run_id, "consistency-control")
        == service.read(state.run_id, "consistency-control")
        == envelope
    )
    # The graph reader may retain valid local facts without an aggregate evaluation.
    graph_read = service.graph_reader.read
    with monkeypatch.context() as local_reader:

        def local_facts(run_id):
            graph, _, facts, report, synthetic = graph_read(run_id)
            return graph, None, facts, report, synthetic

        local_reader.setattr(service.graph_reader, "read", local_facts)
        local = service.save(state.run_id, "local-control")
        assert local.graph_status == "unavailable" and local.accepted_validations
        assert not any(
            e.value is not None and e.source.qualification == "accepted_graph"
            for e in local.entries
        )
        assert service.read(state.run_id, "local-control") == local
    observation = service.observations[0]
    first = observation.outputs[0]
    measurement = first[1].model_copy(
        update=dict(layer="measurement", metric="measurement", unit="px2", direction="none")
    )
    conflicts = (
        (first[0], measurement, "png_foreground_pixels/1.0.0", "value"),
        (first[0], first[1], "scalar-score-output/1.0.0", "value"),
        (first[0], first[1], first[2], "other_field"),
        (
            first[0],
            first[1].model_copy(update=dict(definition="Different meaning")),
            first[2],
            first[3],
        ),
    )
    before = counts(service.runtime)
    audit_before = len(p.audit_reader.list_correlation("SOURCE:DETERMINISTIC_PIPELINE"))
    objects_before = {
        str(f.relative_to(p.objects.root)) for f in p.objects.root.rglob("*") if f.is_file()
    }
    stage_calls = []

    def stage_forbidden(*args):
        stage_calls.append(True)
        raise AssertionError("invalid score reached stage")

    with monkeypatch.context() as m:
        m.setattr(p.objects, "stage", stage_forbidden)
        for conflict in conflicts:
            for outputs in ((first, conflict), (conflict, first)):
                observations = (replace(observation, outputs=outputs),)
                with pytest.raises(ValueError, match="score_definition_conflict"):
                    CaseScoreService(
                        runtime=service.runtime,
                        executions=service.executions,
                        objects=p.objects,
                        observations=observations,
                        synthetic=True,
                    )
                # Recheck after construction, before object staging or database writes.
                with monkeypatch.context() as drift:
                    drift.setattr(service, "observations", observations)
                    with pytest.raises(ValueError, match="score_definition_conflict"):
                        service.save(state.run_id, "conflicting-registry")
        # A change during the controlled source read must also fail before stage.
        with monkeypatch.context() as during_read:

            def changed_registry(run_id):
                result = graph_read(run_id)
                during_read.setattr(
                    service, "observations", (replace(observation, outputs=(first, conflicts[0])),)
                )
                return result

            during_read.setattr(service.graph_reader, "read", changed_registry)
            with pytest.raises(ValueError, match="score_definition_conflict"):
                service.save(state.run_id, "registry-changed-during-read")
        # Distinct assembly scopes that recover the same Artifact still cannot
        # bypass the runtime binding conflict check. Recovery uses actual PG facts.
        recovered = service.executions.recover(state.run_id, observation.key)
        recover = service.executions.recover
        alias = replace(observation, key="synthetic-alias", outputs=(conflicts[0],))
        with monkeypatch.context() as aliases:
            aliases.setattr(
                service.executions,
                "recover",
                lambda run, key: recovered if key == alias.key else recover(run, key),
            )
            for declarations in ((observation, alias), (alias, observation)):
                conflicting = CaseScoreService(
                    runtime=service.runtime,
                    executions=service.executions,
                    objects=p.objects,
                    observations=declarations,
                    synthetic=True,
                )
                with pytest.raises(ValueError, match="score_definition_conflict"):
                    conflicting.save(state.run_id, "artifact-binding-conflict")
        for change in MUTATIONS:
            forged = ScoreEnvelope.model_construct(**mutate(envelope, change))
            with monkeypatch.context() as forged_service:
                forged_service.setattr(service, "build", lambda _, value=forged: value)
                with pytest.raises(ValueError, match=MUTATIONS[change]):
                    service.save(state.run_id, "invalid-" + change)
                with pytest.raises(ValueError, match=MUTATIONS[change]):
                    service.read(state.run_id, "consistency-control")
            # Repository JSON path with correct object checksum still rejects the forged model.
            from packages.dataset_adapters.models import ObjectReference
            from packages.object_store import checksum

            raw = canonical(forged)
            ref = ObjectReference(
                uri="object://knowledge/objects/" + "a" * 32,
                checksum=checksum(raw),
                byte_size=len(raw),
            )
            row = dict(
                operation="case_score_ledger",
                request_hash=digest(
                    dict(run_id=state.run_id, policy_checksum=service.policy_checksum)
                ),
                response_payload=ref.model_dump(mode="json"),
            )
            with monkeypatch.context() as stored:
                stored.setattr(p.objects, "read_bounded", lambda *args, value=raw: value)
                with pytest.raises(ValueError, match=MUTATIONS[change]):
                    service.runtime._score_read(
                        row, state.run_id, service.policy_checksum, lambda: envelope, p.objects
                    )
    assert not stage_calls
    assert counts(service.runtime) == before
    assert len(p.audit_reader.list_correlation("SOURCE:DETERMINISTIC_PIPELINE")) == audit_before
    assert {
        str(f.relative_to(p.objects.root)) for f in p.objects.root.rglob("*") if f.is_file()
    } == objects_before
    assert service.read(state.run_id, "consistency-control") == envelope

    # Identical declarations, including missing outputs, merge before policy hashing/build.
    for key in (observation.key, "not-executed"):
        single = replace(observation, key=key)
        duplicate = replace(single, outputs=single.outputs * 2)
        services = [
            CaseScoreService(
                runtime=service.runtime,
                executions=service.executions,
                objects=p.objects,
                observations=(o,),
                synthetic=True,
            )
            for o in (single, duplicate)
        ]
        assert services[0].policy_checksum == services[1].policy_checksum
        assert services[0].build(state.run_id) == services[1].build(state.run_id)


@pytest.mark.parametrize("change", ["field", "decoder", "unit", "meaning", "range", "direction"])
def test_binding_conflicts_are_rejected_before_any_plan_or_runtime_read(change):
    from packages.case_score_ledger.models import ScoreDefinition
    from packages.case_score_ledger.service import ToolObservation

    d = ScoreDefinition(
        code="geometry",
        layer="measurement",
        metric="measurement",
        definition="Synthetic width",
        unit="px",
        direction="none",
    )
    first = ("ARTIFACT:SYNTHETIC", d, "disc-geometry/1.0.0", "width_px")
    updates = {
        "unit": {"unit": "mm"},
        "meaning": {"definition": "Another meaning"},
        "range": {"maximum": 10.0},
        "direction": {"direction": "higher"},
    }
    second = (
        first[0],
        d.model_copy(update=updates.get(change, {})),
        "scalar-score-output/1.0.0" if change == "decoder" else first[2],
        "height_px" if change == "field" else first[3],
    )
    for outputs in ((first, second), (second, first)):
        with pytest.raises(ValueError, match="score_definition_conflict"):
            CaseScoreService(
                runtime=None,
                executions=None,
                objects=None,
                synthetic=True,
                observations=(ToolObservation("synthetic", "unreadable", None, outputs),),
            )


def test_assembly_sets_have_canonical_order_but_distinct_output_scopes():
    from packages.case_score_ledger.service import ToolObservation, normalized_observations
    from packages.contracts import VersionedReference
    from tests.case_score_ledger.test_models import definition as logit

    d = logit()
    first = ("ARTIFACT:FIRST", d, "synthetic_png_logit/1.0.0", "value")
    second = ("ARTIFACT:SECOND", d, "synthetic_png_logit/1.0.0", "value")
    models = tuple(
        VersionedReference(code=f"SOURCE:SYNTHETIC_MODEL_{i}", version="1.0", checksum=digest(i))
        for i in (1, 2)
    )
    original = ToolObservation(
        "main", "unused", None, (first, second), ("a", "b"), ("c", "d"), models
    )
    shuffled = replace(
        original,
        outputs=(second, first, second),
        required_dependencies=("b", "a", "a"),
        optional_dependencies=("d", "c", "c"),
        model_versions=(*reversed(models), models[0]),
    )
    assert normalized_observations((original,)) == normalized_observations((shuffled,))
    assert len(normalized_observations((shuffled,))[0].outputs) == 2
