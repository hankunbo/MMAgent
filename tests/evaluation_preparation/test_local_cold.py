"""Cold 1.74 construction through original context and campaign classes."""

import json
from datetime import datetime, timedelta
from types import SimpleNamespace

from packages.grounding.models import digest
from packages.multimodal_research import campaign_renewal, context_review
from packages.multimodal_research import local_review_recovery as r
from packages.multimodal_research import unstarted_window as w
from tests.evaluation_preparation.test_execution_lease_postgres import (
    runtime_owner as runtime_owner,
)
from tests.evaluation_preparation.test_null_review_presentation import assert_cold_context_assembly
from tests.grounding.test_database import database as database


def test_local_successor_cold_restores_original_review_projection(
    database, runtime_owner, tmp_path, monkeypatch
):
    captured = []
    original_build = context_review.from_verified_parent

    def capture(*args):
        value = original_build(*args)
        captured.append(value)
        return value

    monkeypatch.setattr(context_review, "from_verified_parent", capture)
    # Original repository constructors verify roles against this owned PG.
    # Parent facts remain the existing synthetic acceptance fixture.
    assert_cold_context_assembly(tmp_path, monkeypatch, runtime_owner.target)
    precursor = captured[-1]
    (precursor.output / "protocol.json").write_text(json.dumps(precursor.protocol))
    failed_path = tmp_path / "failed"
    failed_path.mkdir()
    old = w.review_plan_for(
        precursor.protocol,
        previous=precursor.output,
        output=failed_path,
        authorization="test",
        started=datetime.fromisoformat(precursor.protocol["deadline"]) + timedelta(seconds=1),
        implementation={},
    )
    old["previous_attempts"] = [
        dict(directory=str(tmp_path / "baseline"), requests=26, checksum=digest("baseline"))
    ]
    origin = w.build_review(precursor, failed_path, old)
    proof = dict(
        proof=dict(charged_attempts=1),
        result_checksum=digest("failed"),
        phase_checksum=digest("phase"),
    )
    out = tmp_path / "new"
    out.mkdir()
    plan = r.plan_for(
        old,
        proof,
        previous=failed_path,
        output=out,
        authorization="test",
        started=datetime.fromisoformat(old["deadline"]),
        implementation={},
    )
    (out / "protocol.json").write_text(json.dumps(plan))
    docs = SimpleNamespace(
        read=lambda key: (dict(output=str(out), plan=plan),) if key == r.key(old) else None
    )
    monkeypatch.setattr(r, "failure", lambda current: (docs, proof))
    monkeypatch.setattr(campaign_renewal, "assemble", lambda *a, **kw: origin)
    restored = r.assemble(out)
    assert type(restored) is campaign_renewal.RenewedCampaign
    assert restored.parent is origin.parent
    assert restored.config == origin.config and restored.protocol == plan
    assert restored.base.review_display_context == origin.base.review_display_context
    assert restored.base.review_target_ids == origin.base.review_target_ids
    assert restored.base.confirmed_review_reader is None
    assert restored.initial_targets == () and restored.reused_review_count == 0
    assert sum(x["requests"] for x in restored.protocol["previous_attempts"]) == 27
    r.recheck(restored)
