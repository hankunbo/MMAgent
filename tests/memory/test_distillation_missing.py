"""Target availability through original 09/10B and owned PG, plus old 1.2 history."""

import copy
import json
import os
from pathlib import Path

import pytest

from packages.counterfactual.models import Candidate, Intervention
from packages.counterfactual.service import CounterfactualSimulation
from packages.evidence_details.acceptance_models import DomainState
from packages.grounding.models import canonical, digest, validate
from packages.memory.distillation_models import PairData
from packages.memory_distillation import MemoryDistiller, projection
from tests.counterfactual.support import fixture, protection, run
from tests.database.knowledge_support_04a import sql
from tests.memory.support import second_case
from tests.memory.test_distillation import install
from tests.memory.test_relationships import counts


def test_mixed_owner_prepare_save_current_history_and_legacy_rejection(memory, monkeypatch):
    m = memory
    alternative = m.x
    alt_owner, _ = install(m, m.pg, alternative)
    pg = second_case(m.pg, same_case=True)
    # Add a second synthetic task BEFORE original 08 review/acceptance. No saved
    # report or accepted candidate is patched to manufacture the counterexample.
    raw = pg.reader.state.model_dump(mode="json")
    item = copy.deepcopy(raw["policy"]["items"][0])
    item.update(id="item:two", observation_code="OBS:TWO")
    raw["policy"]["items"].append(item)
    raw["policy"]["groups"][0]["slots"].append(["item:two"])
    observation = copy.deepcopy(raw["material"]["observations"][0])
    observation["reference"].update(code="OBS:TWO", checksum=digest("second synthetic observation"))
    raw["material"]["observations"].append(observation)
    raw["material"]["scope"]["policy_checksum"] = digest(raw["policy"])
    raw["material"]["policy_ref"]["checksum"] = digest(raw["policy"])
    raw["material"]["scope"].pop("material_checksum")
    raw["material"]["scope"]["material_checksum"] = digest(raw["material"])
    pg.reader.state = validate(DomainState, raw)
    x = fixture(
        pg=pg,
        allow={"select_check": ("item:one",), "replace_tool": ("item:two",)},
        policy_updates=dict(alternatives={"alt": alternative.fact.checksum}),
    )
    x.service = CounterfactualSimulation(
        reader=x.reader,
        documents=pg.documents,
        policy=x.policy,
        alternative_readers={"alt": alternative.reader},
    )
    candidate = Candidate(
        id="mixed-target-pg",
        interventions=(
            Intervention(kind="select_check", targets=("item:one",), alternative="alt"),
            Intervention(kind="replace_tool", targets=("item:two",)),
        ),
    )
    report = run(x, candidate)["report"]
    assert report["simulated"] is None and report["classification"] == "indeterminate"
    assert report["reasons"] == ["replace_tool:missing_alternative_result"]
    owner, writer = install(m, pg, x)
    x.h.planner.step(pg.binding.run_id, x.h.command("stop"))
    episode = writer.write(
        run_id=pg.binding.run_id,
        request_key="mixed-episode",
        supplemental=(("simulation", candidate.id, "counterfactual_report"),),
    )
    distiller = MemoryDistiller(repository=m.writer)

    def protected():
        return (
            protection(x),
            protection(alternative),
            owner.snapshot(),
            alt_owner.snapshot(),
            sql(
                m.db.admin,
                "SELECT (SELECT count(*) FROM runtime.run),"
                "(SELECT count(*) FROM runtime.tool_attempt),"
                "(SELECT count(*) FROM runtime.tool_step_execution)",
            ),
        )

    before = protected()
    initial = counts(m)
    records = distiller.prepare(request_key="mixed-content", episode_keys=(episode.record.key,))
    assert counts(m) == initial
    pair = records[0]
    assert isinstance(pair.detail, PairData)
    complete, missing = pair.detail.changes
    assert complete.missing_reason is None and complete.simulated_value == 0.3
    assert complete.original_value == 0.3
    assert complete.simulated_observation == alternative.fact.state.material.observations[
        0
    ].reference.model_dump(mode="json")
    assert missing.missing_reason == "missing_alternative_result"
    assert missing.simulated_value is None and missing.simulated_observation is None
    for field in ("baseline", "simulated", "classification", "comparisons", "alternatives"):
        assert getattr(pair.detail, field) == report[field]
    for field in ("reasons", "affected", "remaining"):
        assert getattr(pair.detail, field) == tuple(report[field])

    bad = pair.model_copy(
        update=dict(
            detail=pair.detail.model_copy(
                update=dict(
                    changes=(
                        complete.model_copy(
                            update=dict(missing_reason="missing_observation_or_interpretation")
                        ),
                        missing,
                    )
                )
            )
        )
    )
    with pytest.raises(ValueError, match="^memory_distillation_binding$"):
        m.writer.create_distillation("mixed-content", (bad, *records[1:]))
    assert counts(m) == initial
    saved = m.writer.create_distillation("mixed-content", records)
    after_save = counts(m)
    assert after_save["memory.record"] == initial["memory.record"] + len(records)
    assert after_save["memory.request"] == initial["memory.request"] + 1
    assert after_save["audit.event"] == initial["audit.event"] + 1
    assert distiller.write(request_key="mixed-content", episode_keys=(episode.record.key,)) == saved
    assert counts(m) == after_save
    for record in records:
        current = m.reader.read(record.key)
        history = m.reader.read(record.key, historical=True)
        assert current.record == history.record == record
        assert current.checksum == history.checksum == digest(record)
        assert canonical(history.record) == canonical(record)
        assert current.current and not history.current
        assert not current.eligible_for_retrieval and not current.eligible_for_planning
    assert protected() == before

    # Separate compatibility setup: emulate precisely the OLD 1.2 projection at
    # admission, leaving original reports, SQL, sources, hashes and Audit intact.
    # This is a legacy-history fixture, NOT the valid mixed counterexample above.
    correct_changes = projection.changes

    def old_changes(report, fact, alternatives):
        return tuple(
            change.model_copy(update=dict(missing_reason="missing_observation_or_interpretation"))
            if change.operation == "replace"
            and change.simulated_observation is not None
            and report["simulated"] is None
            else change
            for change in correct_changes(report, fact, alternatives)
        )

    with monkeypatch.context() as patch:
        patch.setattr(projection, "changes", old_changes)
        legacy = distiller.prepare(request_key="legacy-mixed", episode_keys=(episode.record.key,))
        legacy_saved = m.writer.create_distillation("legacy-mixed", legacy)
    frozen = sql(m.db.admin, "SELECT key,checksum,payload::text FROM memory.record ORDER BY key")
    legacy_counts = counts(m)
    for record in legacy:
        history = m.reader.read(record.key, historical=True)
        assert canonical(history.record) == canonical(record) and history.checksum == digest(record)
        with pytest.raises(ValueError, match="^memory_distillation_binding$"):
            m.reader.read(record.key)
    with pytest.raises(ValueError, match="^memory_distillation_binding$"):
        m.writer.create_distillation("legacy-mixed", legacy)
    corrected = distiller.prepare(request_key="legacy-mixed", episode_keys=(episode.record.key,))
    with pytest.raises(ValueError, match="^memory_key_content_conflict$"):
        m.writer.create_distillation("legacy-mixed", corrected)
    assert counts(m) == legacy_counts
    assert (
        sql(m.db.admin, "SELECT key,checksum,payload::text FROM memory.record ORDER BY key")
        == frozen
    )
    assert protected() == before
    evidence = os.environ.get("MMAGENT_10C_R01_EVIDENCE_DIR")
    if evidence:
        output = Path(evidence)
        (output / "mixed-current.json").write_text(saved.model_dump_json(indent=2) + "\n")
        (output / "mixed-pair.json").write_text(pair.model_dump_json(indent=2) + "\n")
        (output / "legacy-history.json").write_text(legacy_saved.model_dump_json(indent=2) + "\n")
        (output / "mixed-counts.json").write_text(
            json.dumps(
                dict(
                    before=initial,
                    after_save=after_save,
                    after_retry=after_save,
                    after_legacy=legacy_counts,
                    owner_and_runtime_unchanged=True,
                ),
                indent=2,
            )
            + "\n"
        )
