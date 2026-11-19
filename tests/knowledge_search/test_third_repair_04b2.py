"""Third repair red: expansion order is not a semantic dimension."""

import pytest

from packages.knowledge_connectors.ports import model_checksum
from packages.knowledge_search.models import (
    ExpansionOutput,
    ExpansionSuggestion,
    ExpansionTrace,
    KnowledgeQueryPlan,
)
from packages.knowledge_search.planning import ExpansionConfig, expand_plan, query_texts
from tests.knowledge_search.support import config, plan


def expanded():
    terms = tuple(
        ExpansionSuggestion(concept_code="CONCEPT:TEST:CONDITION", kind="synonym", term=t)
        for t in ("alpha", "beta")
    )
    cfg = ExpansionConfig(enabled=True, model="fake-model/1", allowlist=terms)

    class Expander:
        def suggest(self, p, c):
            return ExpansionOutput(suggestions=terms).model_dump_json()

    return expand_plan(plan(config()), cfg, Expander())


def test_third_expansion_output_canonical_set():
    p = expanded()
    a = p.expansion.suggestions
    assert model_checksum(ExpansionOutput(suggestions=a)) == model_checksum(
        ExpansionOutput(suggestions=tuple(reversed(a)))
    )


@pytest.mark.parametrize("duplicate", [False, True])
def test_third_trace_semantic_checksum(duplicate):
    p = expanded()
    a = p.expansion.suggestions
    seq = (*a, a[0]) if duplicate else tuple(reversed(a))
    trace = ExpansionTrace.model_validate(
        p.expansion.model_copy(
            update={
                "suggestions": seq,
                "output_checksum": model_checksum(ExpansionOutput(suggestions=seq)),
            }
        )
    )
    changed = KnowledgeQueryPlan.model_validate(p.model_copy(update={"expansion": trace}))
    assert query_texts(p, "europe_pmc") == query_texts(changed, "europe_pmc")
    assert changed.checksum() == p.checksum()


def test_third_snapshot_generators_and_public_schemas():
    import json
    from pathlib import Path

    from db.knowledge import SourceProvider
    from db.knowledge.validation.build import TARGET as OLD
    from db.knowledge.validation.build import artifact as old_artifact
    from db.knowledge.validation.build_v5 import TARGET, artifact
    from packages.knowledge_search import ProviderConfigurationProof

    assert OLD.read_text() == old_artifact()
    assert TARGET.read_text() == artifact()
    for model in (ProviderConfigurationProof, SourceProvider):
        assert (
            json.loads(
                Path(
                    "packages/knowledge_search/schemas/" + model.__name__ + ".schema.json"
                ).read_text()
            )
            == model.model_json_schema()
        )


@pytest.mark.parametrize(
    "mutation", ["adapter", "checksum", "policy", "missing_proofs", "wrong_proofs"]
)
def test_third_identity_and_execution_proof_revalidation(mutation):
    from packages.knowledge_search.models import ExecutionBinding, ProviderIdentity
    from packages.knowledge_search.planning import provider_query
    from tests.knowledge_search.support import policy

    c = config()
    p = plan(c)
    q = provider_query(p, c, policy())
    if mutation in ("missing_proofs", "wrong_proofs"):
        values = {"plan": p, "queries": (q,), "provider_proofs": (q.identity.configuration,)}
        if mutation == "missing_proofs":
            values.pop("provider_proofs")
        else:
            values["provider_proofs"] = (
                q.identity.configuration.model_copy(
                    update={"config_checksum": "sha256:" + "f" * 64}
                ),
            )
        forged = ExecutionBinding.model_construct(**values)
        with pytest.raises(ValueError):
            ExecutionBinding.model_validate(forged)
    else:
        change = {
            "adapter": {"adapter": "crossref"},
            "checksum": {"config_checksum": "sha256:" + "e" * 64},
            "policy": {"policy_version": "9.0.0"},
        }[mutation]
        with pytest.raises(ValueError):
            ProviderIdentity.model_validate(q.identity.model_copy(update=change))


def test_third_cross_provider_hosts_and_typed_metadata(caplog):
    from packages.knowledge_search import PrivacyError, configuration_proof, discovery_url
    from packages.knowledge_search.discovery import identifier, record_id, title
    from tests.knowledge_search.support import policy

    ep = configuration_proof(config(), policy())
    assert discovery_url("https://europepmc.org/article/MED/123", ep)
    assert (
        discovery_url("https://doi.org/10.1000/foo%2Fbar", ep) == "https://doi.org/10.1000/foo/bar"
    )
    with pytest.raises(PrivacyError):
        discovery_url("https://www.nice.org.uk/guidance/ng81", ep)
    for kind, value in [("DOI", "10.1000/abc"), ("PMID", "12345"), ("PMCID", "PMC12345")]:
        assert identifier(kind, value, ep) == value
    assert record_id("MED:12345", ep) == "MED:12345"
    bad = "patient john smith mrn 12345678"
    for call in (
        lambda: title(bad),
        lambda: identifier("PMID", bad, ep),
        lambda: record_id(bad, ep),
    ):
        with pytest.raises(ValueError) as caught:
            call()
        assert bad not in str(caught.value)
    assert bad not in caplog.text


def test_third_expansion_old_checksum_and_legacy_trace_rejected():
    p = expanded()
    raw = p.expansion.model_dump(mode="json")
    raw.pop("suggestion_policy")
    with pytest.raises(ValueError):
        ExpansionTrace.model_validate_json(__import__("json").dumps(raw))
    with pytest.raises(ValueError):
        ExpansionTrace.model_validate(
            p.expansion.model_copy(update={"output_checksum": "sha256:" + "a" * 64})
        )
    seq = tuple(reversed(p.expansion.suggestions))
    trace = ExpansionTrace.model_validate(p.expansion.model_copy(update={"suggestions": seq}))
    assert trace == p.expansion


def test_third_canonical_sealed_metadata_and_privacy_snapshot():
    import json
    import unicodedata

    from db.knowledge import (
        KnowledgeKind,
        KnowledgeReference,
        SearchHit,
        SourceIdentifierType,
        SourceIdentity,
        seal,
    )
    from packages.knowledge_search.discovery import POLICY
    from packages.knowledge_search.models import CandidateHit, HitMetadata
    from packages.knowledge_search.privacy import FORBIDDEN
    from tests.knowledge_search.test_integrity_04b2 import registered_fixture
    from tests.knowledge_search.test_repair_red_04b2 import success

    hit = success().providers[0].hits[0]
    for value in ("Synthetic Ｔitle", "Synthetic &#84;itle", "Synthetic %54itle"):
        with pytest.raises(ValueError):
            CandidateHit.model_validate(hit.model_copy(update={"title": value}))
        ref = KnowledgeReference(
            kind=KnowledgeKind.SEARCH_RUN,
            code="SOURCE:RUN:FIXTURE",
            version="1.0.0",
            checksum="sha256:" + "a" * 64,
        )
        with pytest.raises(ValueError):
            seal(
                SearchHit,
                code="SOURCE:HIT:FIXTURE",
                version="1.0.0",
                run_ref=ref,
                provider_ref=registered_fixture(config()).record.reference(),
                title=value,
                identifiers=tuple(
                    SourceIdentity(
                        identifier_type=SourceIdentifierType(i.identifier_type), value=i.value
                    )
                    for i in hit.identifiers
                ),
                rank=1,
                discovered_at=hit.discovered_at,
                discovery=hit.metadata,
            )
    with pytest.raises(ValueError):
        HitMetadata.model_validate(
            hit.metadata.model_copy(update={"provider_record_id": "MED:%31%32%33"})
        )
    assert POLICY["forbidden_pattern"] == FORBIDDEN.pattern
    assert POLICY["unicode_version"] == unicodedata.unidata_version
    # Unicode category-C ranges are the common Python/SQL canonical wire policy.
    ranges = POLICY["control_ranges"]
    actual = []
    for lo, hi in ranges:
        assert all(unicodedata.category(chr(cp)).startswith("C") for cp in range(lo, hi + 1))
        actual.extend(range(lo, hi + 1))
    assert len(actual) == sum(
        unicodedata.category(chr(cp)).startswith("C") for cp in range(0x110000)
    )
    assert json.dumps(POLICY)
