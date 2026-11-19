"""05C exact Published leaf scan through real PG and official query service."""

import json
from pathlib import Path

import pytest

from db import manage
from db.knowledge.models import ApplicabilityContext
from db.runtime.repository.grounding import RuntimeGroundingLocks
from db.runtime.repository.grounding_query import GroundingQueryService
from packages.grounding import GroundingError, digest
from packages.grounding.query_models import QueryConfig, QueryResult
from tests.disease_packs import support
from tests.grounding.attribution_support import add_producer
from tests.grounding.coverage_support import (
    CLASSIFICATION,
    classify,
    real_draft_inventory,
    summarize,
)
from tests.grounding.query_support import metadata, updated
from tests.grounding.test_database import database as database
from tests.grounding.test_database import key
from tests.grounding.test_models import reseal
from tests.grounding.test_query_runtime import counts, query_request, scoped_mapping
from tests.grounding.test_runtime import create_run
from tests.grounding.test_runtime import integrated as integrated


def test_exact_published_leaf_coverage_and_external_result_reuse(database, integrated, tmp_path):
    data, mapping, writer, catalogs = scoped_mapping(integrated)
    locks = RuntimeGroundingLocks(manage.test_target("runtime"))
    lock = locks.bind(**create_run(database, data, mapping, writer))
    service = GroundingQueryService(locks, writer)
    request = query_request(database, mapping, writer, lock)
    applicability = ApplicabilityContext(
        population=support.context().population,
        conditions=support.context().conditions,
        modalities=support.context().modalities,
    )
    release = database.reader.get_published(
        data["release"].reference(), applicability=applicability
    ).record
    templates, requirements = [], []
    # Enumerate the complete exact release membership, not the subset that has mappings.
    for member in release.member_refs:
        if member.kind == "template":
            templates.append(
                database.reader.get_published(member, applicability=applicability).record
            )
        elif member.kind == "evidence_requirement":
            requirements.append(
                database.reader.get_published_evidence(member, context=support.context()).record
            )
    assert templates and requirements
    rows, results = [], {}
    before = counts(database)
    for template in templates:
        leaf_codes = {
            n.node_code
            for n in template.nodes
            if n.node_type.value == "EVIDENCE"
            and not any(e.target_node_code == n.node_code for e in template.edges)
        }
        scanned = set()
        for requirement in requirements:
            node = database.reader.get_published_evidence(
                requirement.node_ref, context=support.context()
            ).record
            if node.template_ref != template.reference() or node.code not in leaf_codes:
                continue
            scanned.add(node.code)
            assert requirement.reference().model_dump(
                mode="json"
            ) == request.requirement_ref.model_dump(mode="json")
            result = service.query(request)
            assert service.validate_result(request, result) == result
            rows.append(
                dict(
                    template_ref=template.reference().model_dump(mode="json"),
                    requirement_ref=requirement.reference().model_dump(mode="json"),
                    node_ref=node.reference().model_dump(mode="json"),
                    classification=classify(result),
                )
            )
            results["ready"] = result
        assert scanned == leaf_codes, (
            "Published evidence leaf missing from release requirement scan"
        )
    results["missing_data"] = service.query(updated(request, artifacts=[]))
    results["unknown_producer"] = service.query(
        updated(request, artifacts=[], patient_data_types=[])
    )
    results["unknown_environment"] = service.query(updated(request, environment=[]))
    config = reseal(QueryConfig, {**request.config.model_dump(mode="json"), "max_operations": 1})
    results["search_incomplete"] = service.query(updated(request, config=config))
    results["existing_product"] = service.query(
        updated(
            request, artifacts=[metadata(request, mapping.accepted_artifacts[0], "EXISTING05C")]
        )
    )
    catalogs.disable(mapping.candidates[0].implementation_ref)
    results["revoked_or"] = service.query(updated(request, environment=request.environment[1:]))
    catalogs.disable(mapping.candidates[1].implementation_ref)
    results["revoked_all_unknown_environment"] = service.query(updated(request, environment=[]))
    assert not results["revoked_all_unknown_environment"].suggestions
    assert len(results["revoked_or"].candidates) == 1
    assert counts(database) == before and locks.read(lock.run_id) == lock
    for boundary in ("json", "copy", "construct"):
        retained = results["ready"]
        value = (
            json.loads(retained.model_dump_json())
            if boundary == "json"
            else retained.model_copy()
            if boundary == "copy"
            else QueryResult.model_construct(**vars(retained))
        )
        with pytest.raises(GroundingError, match="stale_or_forged"):
            service.validate_result(request, value)
        assert counts(database) == before
    catalogs._disabled.clear()  # Only this owned synthetic fixture.
    # Same-key, different-content retained objects are recomputed, never persisted.
    current_ready = service.query(request)
    forged = reseal(
        QueryResult,
        {
            **current_ready.model_dump(mode="json"),
            "candidates": [current_ready.candidates[0]],
        },
    )
    with pytest.raises(GroundingError, match="stale_or_forged"):
        service.validate_result(request, forged)
    assert counts(database) == before
    baseline_counts_after_queries = counts(database)
    bindings = [lock.model_dump(mode="json")]
    mappings = [mapping.model_dump(mode="json")]
    for mapped in (False, True):
        producer_mapping, catalog = add_producer(
            mapping, catalogs.read_snapshot(mapping.catalog_ref), mapped=mapped
        )
        # Independent immutable versions for two scenario contexts; never rewrite a run.
        version = "1.3.1" if mapped else "1.3.0"
        catalog = reseal(type(catalog), {**catalog.model_dump(mode="json"), "version": version})
        producer_mapping = reseal(
            type(mapping),
            {
                **producer_mapping.model_dump(mode="json"),
                "version": version,
                "catalog_ref": catalog.reference(),
            },
        )
        catalogs.add(catalog)
        writer.new_version(
            mapping.reference(), producer_mapping, support.context(), request_key=key()
        )
        writer.machine_review(producer_mapping.reference(), support.context(), request_key=key())
        other_lock = locks.bind(**create_run(database, data, producer_mapping, writer))
        q = query_request(database, producer_mapping, writer, other_lock)
        producer = catalog.tools[-1]
        q = updated(
            q,
            patient_data_types=[],
            artifacts=[a for a in q.artifacts if a.artifact_ref != producer.outputs[0].artifact_ref]
            + [metadata(q, producer.inputs[0].constraint, "IMAGE05C")],
            environment=[
                e for e in q.environment if e.implementation_ref != producer.implementation_ref
            ],
        )
        checkpoint = counts(database)
        name = "mapped_producer_blocked" if mapped else "mapping_gap"
        results[name] = service.query(q)
        assert service.validate_result(q, results[name]) == results[name]
        assert counts(database) == checkpoint and locks.read(lock.run_id) == lock
        bindings.append(other_lock.model_dump(mode="json"))
        mappings.append(producer_mapping.model_dump(mode="json"))
    assert not results["mapping_gap"].suggestions
    assert results["mapped_producer_blocked"].suggestions
    report = dict(
        schema_version="05c-coverage-report/1.0.0",
        synthetic=True,
        classification_policy=CLASSIFICATION,
        exact_versions=dict(
            release=release.model_dump(mode="json"),
            templates=[t.model_dump(mode="json") for t in templates],
            requirements=[r.model_dump(mode="json") for r in requirements],
            locks=bindings,
            mappings=mappings,
            catalogs=[],
        ),
        published=dict(rows=rows, **summarize(rows, len(templates))),
        scenarios={name: classify(result) for name, result in results.items()},
        results={name: result.model_dump(mode="json") for name, result in results.items()},
        real_draft=real_draft_inventory(Path(__file__).resolve().parents[2]),
        persistence=dict(
            query_writes=False,
            initial_counts=before,
            baseline_counts_after_queries=baseline_counts_after_queries,
            old_lock_unchanged=locks.read(lock.run_id) == lock,
        ),
        boundary="type feasibility only; no real medgraph adapter, model execution or clinical L4",
    )
    # Exact catalog/policy/config sets come from the actual executed requests.
    refs = {r.request.lock.catalog_ref for r in results.values()}
    report["exact_versions"]["catalogs"] = [
        catalogs.read_snapshot(r).model_dump(mode="json") for r in sorted(refs, key=digest)
    ]
    report["exact_versions"]["query_configs"] = [
        r.model_dump(mode="json")
        for r in sorted({r.request.config for r in results.values()}, key=digest)
    ]
    report["checksum"] = digest(report)
    destination = tmp_path / "coverage-report.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("05C_COVERAGE_ARTIFACT=" + str(destination))


def test_coverage_zero_denominator_and_no_duplicate_inflation():
    empty = summarize([], 0)
    assert empty["classified"]["status"] == empty["type_feasible"]["status"] == "not_applicable"
    row = dict(
        template_ref={"checksum": "template"},
        requirement_ref={"checksum": "requirement"},
        classification=dict(categories=["expert_review"], candidate_count=0),
    )
    classified = summarize([row], 1)
    assert classified["classified"]["value"] == 1 and classified["type_feasible"]["value"] == 0
    with pytest.raises(ValueError, match="duplicate"):
        summarize([row, row], 1)
