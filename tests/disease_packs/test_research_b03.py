"""B03 declarations: software fixtures, including simulated real content mode only."""

import pytest

from db.knowledge.models import KnowledgeKind, KnowledgeReference
from packages.disease_packs import DiseasePack, KnowledgeBindings, PackDefinition, seal
from tests.disease_packs.support import declarations


def bindings():
    def ref(kind):
        return KnowledgeReference(
            kind=KnowledgeKind(kind),
            code="SOURCE:FIXTURE",
            version="1.0.0",
            checksum="sha256:" + "a" * 64,
        )

    return KnowledgeBindings(
        release_ref=ref("release"),
        claim_refs=(ref("clinical_claim"),),
        evidence_requirement_refs=(ref("evidence_requirement"),),
        template_ref=ref("template"),
        rule_refs=(ref("validation_rule"),),
        stop_rule_refs=(ref("stop_rule"),),
    )


def test_old_real_research_published_pending_l4_is_rejected():
    _, definition = declarations()
    definition = seal(
        PackDefinition, **(definition.model_dump(exclude={"checksum"}) | {"synthetic": False})
    )
    with pytest.raises(ValueError, match="requires clinical L4"):
        seal(
            DiseasePack,
            definition=definition,
            status="Published",
            clinical_level="pending_L4",
            bindings=bindings(),
            knowledge_gaps=(),
        )


def test_new_research_port_exists():
    from packages.disease_packs import ResearchDiseasePack

    assert ResearchDiseasePack.model_fields["schema_version"].default == "disease-pack/2.0.0"


def research_declaration():
    from packages.disease_packs import ResearchDiseasePack
    from packages.disease_packs.models import definition_reference
    from packages.knowledge_review.research import ResearchScope, context_checksum
    from tests.disease_packs.support import context

    _, d = declarations()
    d = seal(
        PackDefinition,
        **(
            d.model_dump(exclude={"checksum"})
            | {"synthetic": False, "allowed_conclusions": ("insufficient",)}
        ),
    )
    scope = ResearchScope(
        purpose="limited_research_evidence_planning",
        definition_ref=definition_reference(d),
        context_checksum=context_checksum(context()),
        allowed_outputs=("evidence_requirements", "insufficiency", "limitations"),
    )
    return seal(
        ResearchDiseasePack,
        definition=d,
        status="Published",
        clinical_level="pending_L4",
        bindings=bindings(),
        knowledge_gaps=(),
        research_scope=scope,
        research_policy_ref=KnowledgeReference(
            kind=KnowledgeKind.RELEASE_POLICY,
            code="SOURCE:FIXTURE_POLICY",
            version="1.0.0",
            checksum="sha256:" + "b" * 64,
        ),
    )


@pytest.mark.parametrize("field", ["research_scope", "research_policy_ref", "bindings"])
def test_new_declaration_missing_authorization_fields_rejected(field):
    from packages.disease_packs import ResearchDiseasePack

    data = research_declaration().model_dump(exclude={"checksum"})
    data.pop(field)
    with pytest.raises(ValueError):
        seal(ResearchDiseasePack, **data)


@pytest.mark.parametrize("change", [{"clinical_level": "L4"}, {"clinical_level": "synthetic_only"}])
def test_new_declaration_cannot_self_award_l4(change):
    from packages.disease_packs import ResearchDiseasePack

    p = research_declaration()
    with pytest.raises(ValueError):
        seal(ResearchDiseasePack, **(p.model_dump(exclude={"checksum"}) | change))


def test_three_legacy_versions_exact_directory_read_and_replay_bytes(tmp_path):
    from packages.disease_packs import DatasetReference, DiseasePackCatalog, parse_pack

    ds, d = declarations()
    for version, synthetic, status, level in [
        ("1.0.0", False, "Draft", "pending_L4"),
        ("2.0.0", True, "Published", "synthetic_only"),
        ("3.0.0", False, "Published", "L4"),
    ]:
        dataset = seal(
            DatasetReference,
            **(ds.model_dump(exclude={"checksum"}) | {"version": version, "synthetic": synthetic}),
        )
        definition = seal(
            PackDefinition,
            **(
                d.model_dump(exclude={"checksum"})
                | {
                    "version": version,
                    "synthetic": synthetic,
                    "dataset_refs": (dataset.reference(),),
                }
            ),
        )
        pack = seal(
            DiseasePack,
            definition=definition,
            status=status,
            clinical_level=level,
            bindings=bindings() if status == "Published" else None,
            knowledge_gaps=("review pending",) if status == "Draft" else (),
        )
        folder = tmp_path / version
        (folder / "datasets").mkdir(parents=True)
        (folder / "pack.json").write_text(pack.model_dump_json())
        (folder / "datasets" / "ds.json").write_text(dataset.model_dump_json())
        catalog = DiseasePackCatalog.from_directory(folder)
        assert catalog.pack(pack.reference()).model_dump_json() == pack.model_dump_json()
        assert parse_pack(pack.model_dump_json()).checksum == pack.checksum


def test_context_legacy_bytes_and_new_config_scope():
    from db.knowledge.repository import KnowledgeConflictError
    from packages.graph_engine import canonical_checksum
    from packages.knowledge_review.research import KnowledgeUseRequest, validate_use
    from tests.disease_packs.support import context

    p = research_declaration()
    assert "use" not in context().model_dump(mode="json")
    u = KnowledgeUseRequest(
        purpose="limited_research_evidence_planning",
        policy_ref=p.research_policy_ref.to_versioned_reference(),
        scope_checksum=canonical_checksum(p.research_scope),
    )
    from packages.retrieval.knowledge.research import ResearchRetrievalContext

    c = ResearchRetrievalContext(**context().model_dump(), use=u)
    validate_use(p.research_scope, p.research_policy_ref.to_versioned_reference(), c)
    with pytest.raises(KnowledgeConflictError, match="CONTEXT_SCOPE"):
        validate_use(
            p.research_scope,
            p.research_policy_ref.to_versioned_reference(),
            c.model_copy(update={"age_years": 18}),
        )


def test_preparation_snapshot_dispatches_without_upgrading_old_bytes():
    from packages.dataset_adapters.prepared import PreparedSnapshot

    p = research_declaration()
    snapshot = PreparedSnapshot(
        (b"{}", p.model_dump_json().encode()), (), (), b"{}", None, "unknown", "fixture"
    )
    assert snapshot.pack.model_dump_json() == p.model_dump_json()


def test_published_new_pack_directory_keeps_exact_new_version(tmp_path):
    from packages.disease_packs import DatasetReference, DiseasePackCatalog, ResearchDiseasePack
    from packages.disease_packs.models import definition_reference

    p = research_declaration()
    ds, _ = declarations()
    ds = seal(DatasetReference, **(ds.model_dump(exclude={"checksum"}) | {"synthetic": False}))
    definition = seal(
        PackDefinition,
        **(p.definition.model_dump(exclude={"checksum"}) | {"dataset_refs": (ds.reference(),)}),
    )
    scope = p.research_scope.model_copy(update={"definition_ref": definition_reference(definition)})
    p = seal(
        ResearchDiseasePack,
        **(
            p.model_dump(exclude={"checksum"}) | {"definition": definition, "research_scope": scope}
        ),
    )
    (tmp_path / "datasets").mkdir()
    (tmp_path / "pack.json").write_text(p.model_dump_json())
    (tmp_path / "datasets" / "ds.json").write_text(ds.model_dump_json())
    assert DiseasePackCatalog.from_directory(tmp_path).pack(p.reference()) == p


def test_old_context_does_not_accept_new_request_fields():
    from packages.knowledge_review.research import KnowledgeUseRequest
    from packages.retrieval.knowledge.models import RetrievalContext
    from packages.retrieval.knowledge.research import ResearchRetrievalContext
    from tests.disease_packs.support import context

    pack = research_declaration()
    use = KnowledgeUseRequest(
        purpose="limited_research_evidence_planning",
        policy_ref=pack.research_policy_ref.to_versioned_reference(),
        scope_checksum="sha256:" + "a" * 64,
    )
    current = ResearchRetrievalContext(**context().model_dump(), use=use)
    with pytest.raises(ValueError):
        RetrievalContext.model_validate_json(current.model_dump_json())
    with pytest.raises(ValueError):
        ResearchRetrievalContext.model_validate_json(context().model_dump_json())


def test_clinical_port_denies_even_legacy_l4_and_successful_technical_admission():
    from types import SimpleNamespace

    from apps.ophthalmology_disease_packs import DiseasePackService
    from db.knowledge.repository import KnowledgeConflictError

    p = research_declaration()
    legacy = seal(
        DiseasePack,
        definition=p.definition,
        status="Published",
        clinical_level="L4",
        bindings=p.bindings,
        knowledge_gaps=(),
    )
    calls = []

    # Even a successful generic technical admission cannot substitute clinical evaluation.
    def admitted(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(pack=legacy, knowledge_gap=None)

    service = SimpleNamespace(_catalog=SimpleNamespace(pack=lambda ref: legacy), admit=admitted)
    with pytest.raises(KnowledgeConflictError, match="KNO_B03_CLINICAL_DENIED"):
        DiseasePackService.admit_clinical(service, legacy.reference(), "OCT", None, None)
    assert calls == []
