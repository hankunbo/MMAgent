"""No DB/Provider: retain a committed visual phase when cold trial.targets is empty."""

import copy
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from db.runtime.repository.case_pipeline import identity
from packages.grounding.models import digest
from packages.multimodal_research.campaign import MultimodalCampaign
from packages.multimodal_research.stage_failure import saved_progress, snapshot


def fixture(tmp_path, *, target="new-vision-requirement"):
    c = MultimodalCampaign.__new__(MultimodalCampaign)
    c.output, c.campaign_id = tmp_path, "cold-source-history"
    c.protocol = dict(
        deadline=(datetime.now(UTC) + timedelta(minutes=30)).isoformat(),
        max_new_requests=12,
        previous_attempts=[],
    )
    c.base = SimpleNamespace(
        manifest=SimpleNamespace(checksum=digest("input")),
        task=SimpleNamespace(target_code="GOAL:TEST"),
        definition_expansion_policy=dict(index_keys=[], public_queries=["q"], max_operations=12),
    )
    c.config = SimpleNamespace(max_requests=12)

    def rid(t):
        return identity("RUN", "task-research:" + c.request_key(t))

    definition_run, visual_run = rid("definition-expansion"), rid(target)
    phase = dict(
        run_id=visual_run,
        target=target,
        accepted=True,
        evidence_limited=False,
        requests=dict(reservations=1),
        transport=dict(remote_response_confirmed=1),
        error=None,
    )
    revision = dict(rows=[dict(id=target)], added=[target], replaced=[], accepted={}, binding={})

    def receipt(stage, payload):
        return dict(
            schema_version="multimodal-stage-receipt/1.0.0",
            stage=stage,
            binding=digest(c.protocol),
            run_id=definition_run,
            payload=payload,
        )

    saved = {
        definition_run: {
            "dynamic-campaign": (c.protocol,),
            "definition:revision": (revision,),
            "multimodal-stage-operation:requirements": (receipt("requirements", revision),),
            "multimodal-stage-operation:acquisition": (
                receipt("acquisition", dict(phases=[phase])),
            ),
        },
        visual_run: {
            "dynamic-campaign": (c.protocol,),
            "dynamic-phase-result": (phase,),
            "journal:0": (
                dict(
                    schema_version="agent-budget/1.0.0",
                    binding=dict(
                        run_id=visual_run, deadline=c.protocol["deadline"], max_requests=12
                    ),
                ),
            ),
            "journal:1": (dict(phase="reserved", request_id="reserved-vision"),),
        },
    }
    c.contexts = SimpleNamespace(
        read_research_run=lambda run, **kw: object() if run in saved else None
    )
    c.documents = lambda run: SimpleNamespace(run_id=run, read=saved[run].get)
    c.sessions = SimpleNamespace(read=lambda run: SimpleNamespace(phase="stopped", pending=None))
    c.targets = (target,)
    path = c.phase_path(target)
    path.mkdir()
    (path / "request.json").write_text(json.dumps(dict(lock=dict(run_id=visual_run))))
    cold = copy.copy(c)
    cold.targets = ()
    return c, cold, phase


def test_cold_cancel_retains_confirmed_visual_phase(tmp_path):
    hot, cold, phase = fixture(tmp_path)
    assert saved_progress(hot, "review")["confirmed_phases"] == [phase]
    result = saved_progress(cold, "review")
    assert result["confirmed_phases"] == [phase]
    assert result["run_id"] == phase["run_id"]


def test_cold_cancel_counts_original_visual_reservation(tmp_path):
    hot, cold, _ = fixture(tmp_path)
    assert hot.usage() == 1
    assert cold.usage() == 0
    result = snapshot(cold, ValueError("cancelled"), "review")
    assert result["definition_history_reconstructed"]
    assert result["new_requests"] == 1


def test_failed_scope_before_medical_stages_preserves_original_request_and_control(
    tmp_path, monkeypatch
):
    from packages.multimodal_research import stages

    _, cold, phase = fixture(tmp_path, target="input-scope:synthetic")
    cold.scope_targets = (phase["target"],)
    # Scope executes before medical requirements/acquisition stage receipts exist.
    monkeypatch.setattr(stages, "durable_receipt", lambda *args: None)
    phase.update(accepted=False, error="candidate_failed_or_context_changed")
    result = saved_progress(cold, "acquisition")
    assert result["phase_reservations"] == 1
    assert result["confirmed_phases"] == [phase]
    assert result["phase_controls"][0]["run_id"] == phase["run_id"]
    assert result["phase_controls"][0]["exists"]
    assert result["scores"] is None
