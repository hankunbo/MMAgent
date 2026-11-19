"""Small synthetic 04B7 fixture, all pipeline operations use public services."""

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from db.knowledge import models as m
from db.knowledge.repository.review import admit, bind_artifact, detect, resolve
from packages.contracts import GECEGTemplate, NodeType, ValidationScope
from packages.knowledge_archive import Bibliography
from packages.knowledge_claims.ports import FakeStructuredExtractor
from packages.knowledge_index.models import Filters
from packages.retrieval.knowledge.scope import ClinicalScope
from tests.database.test_knowledge_archive_04b3 import fetch, hit
from tests.database.test_knowledge_claims_04b5 import budget
from tests.database.test_knowledge_release_04b6 import policy, publish, review
from tests.knowledge_claims.support import output


def code(prefix="SOURCE"):
    return prefix + ":B7:" + uuid4().hex.upper()


def make_claim(
    db, store, cfg, *, scope=None, anatomy=(), text="OCT result is positive.", embedder=None
):
    h = hit(db, title="Synthetic retrieval " + uuid4().hex)
    receipt = db.editor.archive_hit(
        h.reference(),
        fetch(
            h,
            body=(
                "<article><body><p>" + text + "</p></body></article><!--" + uuid4().hex + "-->"
            ).encode(),
            bibliography=Bibliography(language="en", publication_date="2024-06-15"),
        ),
        store,
    )
    condition = m.seal(
        m.ApplicabilityCondition,
        code=code(),
        version="1.0.0",
        synthetic=True,
        population=("synthetic_fixture_only",),
        conditions=("software_test",),
        modalities=("synthetic",),
        anatomy=anatomy,
        scope_note="Synthetic fixture only",
        retrieval_scope=scope,
    )
    db.editor.put(condition)
    build = db.editor.build_document_index(
        receipt,
        store,
        cfg,
        embedder=embedder,
        applicability_refs=(condition.reference(),),
        targets=("synthetic-target",),
    )
    filters = Filters(
        population=("synthetic_fixture_only",),
        conditions=("software_test",),
        modalities=("synthetic",),
        anatomy=anatomy,
    )
    from packages.knowledge_index.ports import FakeReranker

    reranker = FakeReranker() if cfg.reranker_mode == "enabled" else None
    indexed = db.editor.query_candidate_index(
        build["key"], "OCT", filters, embedder=embedder, reranker=reranker
    )["hits"][0]
    data = json.loads(output(text))
    if text == "OCT result is negative.":
        data["candidates"][0]["polarity"] = "NEGATIVE"
    extracted = db.editor.extract_claims(
        indexed,
        store,
        FakeStructuredExtractor(json.dumps(data).encode()),
        budget(),
        filters=filters,
    )
    assert extracted["status"] == "succeeded", extracted
    claim = next(s.record for s in extracted["records"] if type(s.record) is m.ClinicalClaim)
    admit(db.editor, claim.reference(), indexed, store, filters=filters)
    return claim, indexed, build, condition


def assemble(
    db, store, cfg, *, scoped=False, conflict=False, graph=False, embedder=None, negative=False
):
    scope = (
        ClinicalScope(
            targets=("synthetic-target",),
            disease_packs=("synthetic-pack",),
            sexes=("female",),
            exams=("OCT",),
            times=("baseline",),
            lateralities=("RIGHT",),
            devices=("synthetic-device",),
            age_min=18,
            age_max=65,
        )
        if scoped
        else None
    )
    c, h, build, condition = make_claim(
        db,
        store,
        cfg,
        scope=scope,
        anatomy=("synthetic_retina",) if scoped else (),
        embedder=embedder,
        text="OCT result is negative." if negative else "OCT result is positive.",
    )
    p = policy(db)
    other = None
    if conflict:
        other, other_hit, _, _ = make_claim(
            db, store, cfg, anatomy=("other_anatomy",), embedder=embedder
        )
        conflicts = detect(db.editor, (c.reference(), other.reference()))
        assert conflicts[0].record.conflict_type == m.ConflictType.ANATOMY
        cr = review(db, conflicts[0].record, p, disposition="conditional_coexist", transition=False)
        resolve(db.editor, conflicts[0].record.reference(), cr.reference(), version="1.1.0")
    cr = review(db, c, p)
    publish(db, store, c, cr, p)
    common = dict(
        version="1.0.0",
        synthetic=True,
        source_refs=c.source_refs,
        applicability_refs=c.applicability_refs,
    )
    members = [c.reference()]
    rule = template = None
    if graph:
        rule = bind_artifact(
            db.editor,
            m.seal(
                m.ValidationRule,
                code=code("RULE"),
                rule_scope=ValidationScope.EVIDENCE,
                expression_code="RULE:SYNTHETIC_ASSERTION",
                claim_refs=(c.reference(),),
                **common,
            ),
            ((c.reference(), cr.reference()),),
            p.reference(),
        )
        db.editor.put(rule)
        rr = review(db, rule, p)
        publish(db, store, rule, rr, p)
        concept = m.seal(
            m.Concept, code=code("CONCEPT"), name="Synthetic software target", **common
        )
        db.editor.put(concept)
        node = m.TemplateNode(
            node_code=code("GOAL"),
            node_type=NodeType.GOAL,
            concept_code=concept.code,
            required=True,
        )
        template_code = code("TEMPLATE")
        base = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "packages/contracts/examples/v1/GECEGTemplate.valid.json"
            ).read_text()
        )
        base.update(
            template_code=template_code,
            template_version="1.0.0",
            nodes=[node.model_dump(mode="json")],
            edges=[],
            goal_node_code=node.node_code,
            citations=[r.to_versioned_reference().model_dump(mode="json") for r in c.source_refs],
        )
        base["provenance"].update(
            created_at=datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
            source_codes=[c.source_refs[0].code],
            rule_versions=[rule.reference().to_versioned_reference().model_dump(mode="json")],
            knowledge_versions=[c.reference().to_versioned_reference().model_dump(mode="json")],
        )
        public = GECEGTemplate.model_validate_json(json.dumps(base))
        template = bind_artifact(
            db.editor,
            m.seal(
                m.Template,
                code=template_code,
                goal_node_code=node.node_code,
                nodes=(node,),
                edges=(),
                rule_refs=(rule.reference(),),
                graph_template=public,
                **common,
            ),
            ((c.reference(), cr.reference()),),
            p.reference(),
        )
        db.editor.put(template)
        n = bind_artifact(
            db.editor,
            m.seal(
                m.Node,
                code=node.node_code,
                template_ref=template.reference(),
                concept_ref=concept.reference(),
                definition=node,
                **common,
            ),
            ((c.reference(), cr.reference()),),
            p.reference(),
        )
        db.editor.put(n)
        tr = review(db, template, p)
        publish(db, store, template, tr, p)
        members.extend((rule.reference(), template.reference()))
    release = bind_artifact(
        db.editor,
        m.seal(
            m.Release, code=code(), policy_ref=p.reference(), member_refs=tuple(members), **common
        ),
        ((c.reference(), cr.reference()),),
        p.reference(),
    )
    db.editor.put(release)
    rr = review(db, release, p)
    release_key = uuid4().hex
    publish(db, store, release, rr, p, release_key)
    return dict(
        release=release,
        claim=c,
        hit=h,
        build=build,
        condition=condition,
        policy=p,
        release_review=rr,
        release_key=release_key,
        rule=rule,
        template=template,
        other=other,
    )
