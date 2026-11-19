"""Dynamic evidence inventory and exact partial admission, synthetic requests only."""

import pytest

from packages.evidence_details.acceptance_models import DomainState, TaskPolicyV3
from packages.evidence_details.review_batches import aggregate, collect_partial
from packages.evidence_scoring.accepted import build_accepted
from packages.evidence_scoring.dynamic_policy import POLICY
from packages.grounding.models import digest, validate
from tests.agent_provider.batch_support import batch_reply, fixture, generate


def dynamic_fixture(count=11, *, empty=False, replies=None, config_path=None):
    f = fixture(count, replies=replies, **({"config_path": config_path} if config_path else {}))
    raw = f.owner.state.model_dump(mode="json")
    old = raw["policy"]
    available = {
        o["reference"]["code"] for o in raw["material"]["observations"] if o["status"] == "obtained"
    }
    requirements = [
        dict(
            id=i["id"],
            definition=i["definition"],
            input_modalities=["SLO"],
            basis_kind="qualitative",
        )
        for i in old["items"]
    ]
    policy = validate(
        TaskPolicyV3,
        dict(
            **{
                k: v
                for k, v in old.items()
                if k not in {"schema_version", "items", "aggregation_checksum"}
            },
            aggregation_checksum=digest(POLICY),
            items=[] if empty else [i for i in old["items"] if i["observation_code"] in available],
            requirements=requirements,
        ),
    )
    material = raw["material"]
    material["observations"] = [
        o for o in material["observations"] if not empty and o["reference"]["code"] in available
    ]
    material["scope"]["policy_checksum"] = digest(policy)
    material["policy_ref"]["checksum"] = digest(policy)
    material["scope"].pop("material_checksum")
    material["scope"]["material_checksum"] = digest(material)
    raw.update(policy=policy, material=material)
    f.owner.state = validate(DomainState, raw)
    return f


def test_dynamic_observations_not_fixed_slots_and_unchanged_denominator():
    f = dynamic_fixture()
    state = f.owner.current()
    assert len(state.policy.requirements) == 11
    assert len(state.policy.items) == 5
    assert all(i.role != "model" for i in state.policy.items)
    submission = generate(f)
    f.acceptance.submit(submission, key="full")
    receipt = f.acceptance.accept(key="full")
    assert receipt.status == "accepted"
    scores = f.acceptance.read(key="full")[2]["scores"]
    assert scores["schema_version"] == "case-research-scoring/7.0.0"
    assert scores["root"]["coverage"]["slo_image_research"]["acquisition"] == 5 / 11
    assert scores["review_completion"]["pending_ids"] == []


def test_zero_observations_normal_no_request_no_invented_disease_score():
    f = dynamic_fixture(empty=True)
    value = build_accepted(f.owner.current(), None)
    assert value["reviews"] == []
    assert value["root"]["summary"]["confidence"]["value"] is None
    assert value["root"]["coverage"]["slo_image_research"]["acquisition"] == 0
    assert f.budget.records() == () or f.budget.records() == []


def test_partial_acceptance_requires_exact_original_receipts_and_preserves_missing():
    from packages.llm_provider.provider import ProviderError
    from packages.llm_provider.transport import Reply

    f = dynamic_fixture(replies=[batch_reply, Reply(content=b'{"partial":', error="timeout")])
    with pytest.raises(ProviderError, match="timeout"):
        generate(f)
    partial = collect_partial(f.owner.current(), f.documents, "synthetic-batches")
    assert partial.pending_batch_ids == ("batch:001",)
    assert f.acceptance.submit(partial, key="partial")["status"] == "submitted_not_accepted"
    assert f.acceptance.accept(key="partial").status == "accepted"
    saved = f.acceptance.read(key="partial")
    assert saved[0]["schema_version"] == "case-observation-interpretation/2.4.0"
    scores = saved[2]["scores"]
    assert len(scores["reviews"]) == 3
    assert len(scores["review_completion"]["pending_ids"]) == 2
    assert scores["root"]["coverage"]["slo_image_research"]["acquisition"] == 5 / 11
    n = len(f.documents.audit)
    assert f.acceptance.accept(key="partial").status == "accepted"
    assert len(f.documents.audit) == n
    from agents.expert_agent.reporting import read_research_report, render_research_html

    report = read_research_report(f.acceptance, key="partial", parent=f.owner.current())
    assert "部分批次已接纳" in render_research_html(report)
    forged = partial.model_copy(update={"pending_batch_ids": ("batch:000",)})
    assert f.acceptance.submit(forged, key="forged").status == "rejected"


def test_partial_cannot_cherry_pick_already_confirmed_batches():
    f = dynamic_fixture()
    full = generate(f)
    partial = aggregate(f.owner.current(), full.plan, full.batches[:1], partial=True)
    assert f.acceptance.submit(partial, key="omit-completed").status == "rejected"


def test_old_contract_cannot_silently_accept_partial():
    f = fixture(7)
    full = generate(f)
    with pytest.raises(ValueError, match="partial_review_requires_dynamic_policy"):
        aggregate(f.owner.current(), full.plan, full.batches[:1], partial=True)


def test_dynamic_does_not_allow_unknown_patient_observation():
    f = dynamic_fixture()
    raw = f.owner.state.model_dump(mode="json")
    raw["material"]["observations"][0]["status"] = "unknown"
    raw["material"]["observations"][0]["value"] = None
    raw["material"]["scope"].pop("material_checksum")
    raw["material"]["scope"]["material_checksum"] = digest(raw["material"])
    with pytest.raises(ValueError, match="dynamic_evidence_requires_observation"):
        validate(DomainState, raw)


@pytest.mark.parametrize("direction", [0.35, 0.0, -0.35])
def test_qualitative_relation_without_numeric_threshold_is_scored(direction):
    def reply(body):
        raw = batch_reply(body)
        for row in raw["items"]:
            if row["status"] == "assessed":
                row["judgment"].update(
                    value=direction,
                    lower=direction,
                    upper=direction,
                    reason="Synthetic qualitative comparison, not a literature effect size.",
                )
        return raw

    f = dynamic_fixture(1, replies=[reply])
    raw = f.owner.state.model_dump(mode="json")
    source = raw["material"]["sources"][0]
    source["text"] = "Synthetic qualitative fixture: irregular edges support the target pattern."
    source["reference"]["checksum"] = digest(source["text"])
    observation = raw["material"]["observations"][0]
    observation["value"] = "Synthetic observer sees a mildly irregular edge."
    observation["reference"]["checksum"] = digest(observation["value"])
    definition = raw["policy"]["items"][0]["definition"]
    definition.update(
        examination="Synthetic edge appearance",
        interpretation="Irregular edges support the synthetic target; no medical threshold.",
        source_refs=[source["reference"]],
    )
    definition["reference"]["checksum"] = digest(
        {k: v for k, v in definition.items() if k != "reference"}
    )
    raw["policy"]["requirements"][0]["definition"] = definition
    raw["material"]["scope"]["policy_checksum"] = digest(raw["policy"])
    raw["material"]["policy_ref"]["checksum"] = digest(raw["policy"])
    raw["material"]["scope"].pop("material_checksum")
    raw["material"]["scope"]["material_checksum"] = digest(raw["material"])
    f.owner.state = validate(DomainState, raw)
    assert f.owner.current().policy.requirements[0].basis_kind == "qualitative"
    assert "no medical threshold" in f.owner.current().policy.items[0].definition.interpretation
    proposal = generate(f)
    f.acceptance.submit(proposal, key="qualitative")
    assert f.acceptance.accept(key="qualitative").status == "accepted"
    score = f.acceptance.read(key="qualitative")[2]["scores"]
    row = score["reviews"][0]
    assert row["judgment"]["value"] == direction
    assert row["weight"] > 0
    assert row["support"] == pytest.approx(row["weight"] * max(direction, 0))
    assert row["refutation"] == pytest.approx(row["weight"] * max(-direction, 0))
    assert score["root"]["coverage"]["slo_image_research"]["interpretation"] == 1


def test_dynamic_still_rejects_unrelated_reference():
    def reply(body):
        raw = batch_reply(body)
        raw["items"][0]["reference_citations"] = ["SOURCE:NOT_IN_TASK"]
        return raw

    from packages.llm_provider.provider import ProviderError

    f = dynamic_fixture(1, replies=[reply])
    with pytest.raises(ProviderError, match="scope"):
        generate(f)
    assert f.documents.read("research-acceptance:unrelated-reference") is None
