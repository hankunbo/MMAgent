"""No direction/score expectations: qualify only current unreviewed definition work."""

from copy import deepcopy

import pytest

from packages.multimodal_research import campaign_renewal as c
from packages.multimodal_research import definition_alternative as d
from tests.evaluation_preparation.test_alternative_acquisition import plan
from tests.evaluation_preparation.test_exact_acquisition_budget import configuration


def terminal_fixture(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace as NS

    from packages.grounding.models import digest

    child, parent_path = tmp_path / "child", tmp_path / "parent"
    child.mkdir()
    parent_path.mkdir()
    old = {"complete": True, "origin": "admitted-definition"}
    result = {"complete": True, "phases": [{"run_id": "review"}]}
    plan = dict(
        schema_version=d.VERSION,
        facts=str(parent_path),
        original_fact_result=digest(old),
        tip={"run_id": "source"},
    )
    (parent_path / "result.json").write_text(json.dumps(old))
    (child / "protocol.json").write_text(json.dumps(plan))
    (child / "result.json").write_text(json.dumps(result))
    claim = {"output": str(child), "plan_checksum": digest(plan)}
    parent = NS(output=parent_path, documents=lambda _: NS(read=lambda key: (claim,)))
    restored = NS(documents=lambda _: NS(read=lambda key: (result,)))
    original = c.assemble
    seen = []

    def dispatch(path, **kwargs):
        if str(path) == str(parent_path):
            seen.append(kwargs)
            return parent
        return original(path, **kwargs)

    monkeypatch.setattr(c, "assemble", dispatch)
    monkeypatch.setattr(c, "RenewedCampaign", lambda *args: restored)
    monkeypatch.setattr(c, "stopped_tip", lambda *args, **kwargs: {})
    return child, plan, claim, restored, result, seen


def test_completed_definition_alternative_is_not_its_own_ancestor_cycle(tmp_path, monkeypatch):
    child, _, _, restored, result, seen = terminal_fixture(tmp_path, monkeypatch)
    assert c.completed_parent(child) == (restored, result)
    assert seen == [{"depth": 2, "ancestry": (child,)}]


def test_definition_alternative_direct_restore_and_claim_gate(tmp_path, monkeypatch):
    child, _, claim, restored, _, seen = terminal_fixture(tmp_path, monkeypatch)
    assert c.assemble(child) is restored
    assert seen == [{"depth": 1, "ancestry": (child,)}]
    claim["output"] = str(tmp_path / "wrong-child")
    with pytest.raises(ValueError, match="renewal_claim_binding"):
        c.completed_parent(child)


def test_definition_alternative_actual_cycle_and_depth_still_reject(tmp_path, monkeypatch):
    import json

    child, plan, _, _, _, _ = terminal_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="cycle"):
        c.assemble(child, ancestry=(child, tmp_path / "other"))
    with pytest.raises(ValueError, match="cycle|depth"):
        c.assemble(child, depth=c.MAX_HISTORY_DEPTH + 1)
    plan["facts"] = str(child)
    (child / "protocol.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="cycle"):
        c.assemble(child)


def fixture():
    protocol = dict(mode="definitions", definition_increment_only=True)
    phases = [
        dict(target="new", accepted=True, acceptance_key="vision-original"),
        dict(
            target=None,
            accepted=False,
            acceptance_key=None,
            error=None,
            evidence_limited=True,
            requests=dict(api_requests_observed=0, api_requests_unknown=0),
        ),
    ]
    result = dict(complete=True, phases=phases)
    revision = dict(
        added=[],
        replaced=[dict(id="new")],
        accepted=[dict(id="admission")],
        rows=[
            dict(
                id="new",
                component="image",
                acquisition_method="vision",
                acquisition_allowed=True,
                review_allowed=True,
            )
        ],
    )
    return protocol, result, revision


def test_confirmed_uninterpretable_observation_allows_admitted_requirement():
    protocol, result, revision = fixture()
    assert d.selection(protocol, result, revision, []) == ["new"]
    revision.update(added=["new"], replaced=[])
    assert d.selection(protocol, result, revision, []) == ["new"]


@pytest.mark.parametrize(
    "bad",
    [
        "accepted_review",
        "review_key",
        "unknown",
        "reserved",
        "review_failure",
        "incomplete",
        "unaccepted_vision",
        "two_targets",
        "unadmitted",
        "missing_input",
        "blocked_relation",
    ],
)
def test_no_replacement_or_unknown_replay_or_missing_qualification(bad):
    protocol, result, revision = fixture()
    events = []
    if bad == "accepted_review":
        result["phases"][-1]["accepted"] = True
    elif bad == "review_key":
        result["phases"][-1]["acceptance_key"] = "already-accepted"
    elif bad == "unknown":
        result["phases"][-1]["requests"]["api_requests_unknown"] = 1
    elif bad == "reserved":
        events.append(dict(phase="reserved", request_id="remote-unknown"))
    elif bad == "review_failure":
        result["phases"][-1]["error"] = "transport"
    elif bad == "incomplete":
        result["complete"] = False
    elif bad == "unaccepted_vision":
        result["phases"][0]["accepted"] = False
    elif bad == "two_targets":
        revision["added"].append("other")
    elif bad == "unadmitted":
        revision["accepted"] = []
    elif bad == "missing_input":
        revision["rows"][0]["component"] = None
    elif bad == "blocked_relation":
        revision["rows"][0]["review_allowed"] = False
    with pytest.raises(ValueError, match="definition_alternative_"):
        d.selection(protocol, result, revision, events)


def test_new_version_uses_original_exact_budget_without_inherited_judgments():
    value = plan()
    value.update(
        schema_version=d.VERSION,
        review_context_scope=None,
        definition_increment_only=True,
        reuse_assessment_ids=[],
    )
    assert d.VERSION in c.EXACT_ACQUISITION_VERSIONS
    assert d.VERSION not in c.RETAINED_ACQUISITION_VERSIONS
    c.validate_exact_acquisition_budget(value, configuration())
    assert c.acquisition_review_ids(value) == ("check",)
    changed = deepcopy(value)
    changed["max_new_requests"] = changed["max_model_dispatches"] = 1
    with pytest.raises(ValueError, match="renewal_exact_request_budget_binding"):
        c.validate_exact_acquisition_budget(changed, configuration())


def test_new_cli_exposes_exact_target_and_check_only_without_calling_run(monkeypatch, capsys):
    import sys

    from apps.multimodal_case import main

    received = []
    monkeypatch.setattr(d, "prepare", lambda **kw: received.append(kw) or {"eligible": True})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "multimodal_case",
            "definition-alternative-prepare",
            "--previous",
            "old",
            "--output",
            "new",
            "--target",
            "check",
            "--authorization",
            "test",
            "--check-only",
        ],
    )
    assert main() == 0
    assert received == [
        dict(
            previous="old",
            output="new",
            target="check",
            authorization="test",
            request_limit=3,
            check_only=True,
        )
    ]
    assert '"eligible": true' in capsys.readouterr().out


def preparation_fixture(tmp_path, monkeypatch, documents=None):
    """Only precursor/method selection substituted; claim owner can be real PG."""
    import json
    from pathlib import Path
    from types import SimpleNamespace as NS

    from packages.grounding.models import digest
    from packages.llm_provider.config import read_provider_config
    from packages.multimodal_research import alternative_acquisition as a
    from packages.multimodal_research import implementation, unknown_acquisition
    from tests.evaluation_preparation.test_unscored_acquisition import OLD, Documents

    old = tmp_path / "original"
    old.mkdir()
    (old / "result.json").write_text(json.dumps(dict(complete=True, new_requests=3)))
    docs = Documents() if documents is None else documents
    saved_revision = {"original_current_admission": True}
    revision = digest(saved_revision)
    tip = dict(
        path=str(old),
        run_id="RUN:SYNTHETIC",
        result=digest("result"),
        previous_attempts=[dict(requests=26)],
        definition=dict(revision=revision),
    )
    parent = NS(
        output=old,
        campaign_id="original",
        logical_campaign_id="original-logical",
        protocol=dict(deadline="2026-10-20T01:00:00+00:00"),
        config=read_provider_config(json.loads(Path(OLD).read_bytes())),
        base=NS(
            manifest=NS(checksum=digest("input")),
            task=NS(checksum=digest("task")),
            definition_current=lambda: saved_revision,
        ),
        accepted_observations=lambda: {},
        available_targets=lambda base: ("new",),
        documents=lambda _: docs,
    )
    calls = []

    def material(*args):
        calls.append("transform")
        work = tmp_path / "work"
        (work / "result").mkdir(parents=True)
        (work / "result/view.png").write_bytes(b"synthetic")
        return work, dict(checksum=digest("synthetic"))

    monkeypatch.setattr(c, "assemble", lambda *a, **kw: parent)
    monkeypatch.setattr(c, "stopped_tip", lambda *a, **kw: deepcopy(tip))
    monkeypatch.setattr(c, "gate_selection", lambda *a: ("new",))
    monkeypatch.setattr(
        c,
        "RenewedCampaign",
        lambda *a: NS(
            freeze=lambda: None, base=NS(alternative_view=NS(read_material=lambda: None))
        ),
    )
    monkeypatch.setattr(d, "check", lambda *a: ["new"])
    monkeypatch.setattr(a, "proof", lambda *a, **kw: dict(target="new"))
    monkeypatch.setattr(a, "prepare_material", material)
    monkeypatch.setattr(implementation, "freeze", lambda: {"synthetic": digest("software")})
    monkeypatch.setattr(unknown_acquisition, "check_network", lambda _: None)
    args = dict(previous=old, output=tmp_path / "new", target="new", authorization="synthetic")
    return docs, args, calls


def test_preparation_same_key_recovers_claim_without_transform_or_budget_reset(
    tmp_path, monkeypatch
):
    import json

    docs, args, calls = preparation_fixture(tmp_path, monkeypatch)
    preview = d.prepare(**args, check_only=True)
    assert preview["new_api_requests"] == preview["database_writes"] == 0
    assert not args["output"].exists()
    before = (args["previous"] / "result.json").read_bytes()
    first = d.prepare(**args)
    frozen = (args["output"] / "protocol.json").read_bytes()
    second = d.prepare(**args)
    assert first["status"] == "prepared" and second["status"] == "already_recorded"
    assert calls == ["transform"] and docs.writes == 1
    assert second["previous_requests"] == 26
    assert (args["output"] / "protocol.json").read_bytes() == frozen
    assert json.loads(frozen)["logical_campaign_id"] == "original-logical"
    assert (args["previous"] / "result.json").read_bytes() == before
    with pytest.raises(ValueError, match="already_claimed"):
        d.prepare(**dict(args, output=tmp_path / "fork"))
    with pytest.raises(ValueError, match="preparation_conflict"):
        d.prepare(**dict(args, request_limit=2))


def test_unknown_extra_plan_switch_is_rejected_before_owner_read():
    from types import SimpleNamespace

    with pytest.raises(ValueError, match="definition_alternative_plan_fields"):
        d.check(SimpleNamespace(), {"reuse_confirmed_review": True})


@pytest.mark.parametrize(
    "change", [None, "unknown", "body", "scope", "transport", "duplicate", "failed_final"]
)
def test_confirmed_source_correction_requires_exact_raw_response_and_final_success(change):
    from types import SimpleNamespace as NS

    from packages.grounding.models import digest

    scope = dict(run_id="run", case_code="case", input_checksum="input")
    events, stored = [], {}
    for i in range(2):
        record = dict(
            request_id=f"r{i}",
            invocation_id="r0",
            retry_index=i,
            role="source",
            error="schema" if i == 0 else None,
            simulated=False,
            api_requests=1,
            scope=scope,
            config_checksum="config",
            prompt_version="prompt",
            output_checksum=digest(f"response{i}".encode().hex()),
        )
        events.extend(
            [
                dict(
                    phase="reserved",
                    request_id=f"r{i}",
                    scope_checksum=digest(scope),
                    config_checksum="config",
                ),
                dict(phase="completed", record=record),
                dict(phase="response_confirmation", request_id=f"r{i}", confirmed=True),
            ]
        )
        stored[f"definition:response:r{i}"] = (dict(record=record, raw_text=f"response{i}"),)
    proposal = {"synthetic_proposal": "accepted by original owner"}
    events[-2]["record"]["output_checksum"] = digest(proposal)
    stored["definition:operation:0"] = (dict(kind="llm", key="definition:proposal"),)
    stored["definition:confirmed-proposal"] = (proposal,)
    if change == "unknown":
        events[-1]["confirmed"] = None
    elif change == "body":
        stored["definition:response:r0"][0]["raw_text"] = "changed"
    elif change == "scope":
        scope["input_checksum"] = "changed"
    elif change == "transport":
        events[1]["record"]["error"] = "transport"
    elif change == "duplicate":
        events.append(events[0])
    elif change == "failed_final":
        events[-2]["record"]["error"] = "schema"
    owner = NS(base=NS(binding=dict(case_code="case")))
    docs = NS(run_id="run", read=stored.get, read_many=lambda ks: tuple(stored.get(k) for k in ks))
    protocol = dict(input="input", expansion_policy=dict(prompt_version="prompt", max_operations=4))
    if change:
        with pytest.raises(ValueError, match="definition_alternative_source_response"):
            d.confirmed_sources(owner, docs, events, protocol)
    else:
        assert d.confirmed_sources(owner, docs, events, protocol).startswith("sha256:")


@pytest.mark.parametrize("changed", [None, "source", "revision"])
def test_repeated_origin_restore_revalidates_original_owner(tmp_path, changed):
    import json
    from types import SimpleNamespace as NS

    from packages.grounding.models import digest
    from packages.multimodal_research.definition_expansion import DefinitionExpansion

    protocol = {"mode": "definitions"}
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    revision = {"rows": [{"id": "revised"}]}
    base = NS(ledger=revision["rows"])
    owner = object.__new__(DefinitionExpansion)
    owner.campaign = NS(base=base, protocol=protocol)
    owner.documents = NS(run_id="original-source")
    calls = []

    # Preserve a bound owner method, as production restoration does.
    class Reader(DefinitionExpansion):
        def read_revision(self):
            calls.append("read-current-source")
            if changed == "source":
                raise ValueError("definition_revision_source_changed")
            return {"rows": []} if changed == "revision" else revision

        def install(self, value):
            calls.append("install")
            base.ledger = value["rows"]
            base.definition_current = self.read_revision

    owner.__class__ = Reader
    base.definition_current = owner.read_revision
    parent = NS(base=base)
    tip = dict(
        path=str(tmp_path), definition=dict(run_id="original-source", revision=digest(revision))
    )
    if changed:
        with pytest.raises(ValueError, match="revision.*changed"):
            c.restore_origin_materials(parent, tip)
        assert calls == ["read-current-source"]
    else:
        c.restore_origin_materials(parent, tip)
        c.restore_origin_materials(parent, tip)
        assert calls == ["read-current-source", "install"] * 2
        assert base.ledger == revision["rows"]


def test_definition_successor_copy_restores_original_baseline_not_revised_parent():
    from types import SimpleNamespace as NS

    from packages.multimodal_research.definition_expansion import DefinitionExpansion

    base = NS(ledger=[{"id": "revised"}], confirmed_review_reader="old-review")
    owner = object.__new__(DefinitionExpansion)
    owner.campaign = NS(base=base)
    owner.original_rows = [{"id": "original"}]
    base.definition_current = owner.read_revision
    copied = c.definition_phase_base(base, original_requirements=True)
    assert copied is not base and copied.ledger == owner.original_rows
    assert copied.ledger is not owner.original_rows
    assert base.ledger == [{"id": "revised"}]
    assert base.confirmed_review_reader == "old-review"
    assert copied.confirmed_review_reader is None
    assert c.definition_phase_base(base).ledger == base.ledger
    with pytest.raises(ValueError, match="original_owner_missing"):
        c.definition_phase_base(NS(ledger=[]), original_requirements=True)
