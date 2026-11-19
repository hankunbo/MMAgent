import copy
from types import SimpleNamespace

from packages.evidence_details.acceptance_models import DomainState, TaskPolicyV2
from packages.grounding.models import digest, validate
from packages.llm_provider.models import Material
from packages.path_graph_validation import PathGraphValidation, ResearchPathReader
from tests.agent_provider.support import review_from_wire, submission
from tests.expert_agent.support import setup


def configured(
    *,
    statuses=("obtained",),
    roles=None,
    targets=None,
    same_source=True,
    shared_product=False,
    alternative_slot=False,
):
    base = setup()
    reader, binding = base.reader, base.binding
    state = reader.state
    policy = state.policy.model_dump(mode="json")
    proto = policy["items"][0]
    material = state.material.model_dump(mode="json")
    obs = material["observations"][0]
    policy["items"], material["observations"] = [], []
    roles = roles or ("visual",) * len(statuses)
    targets = targets or ("GOAL:SYNTHETIC",) * len(statuses)
    for n, (status, role, target) in enumerate(zip(statuses, roles, targets, strict=True)):
        i, o = copy.deepcopy(proto), copy.deepcopy(obs)
        i.update(id=f"item:{n}", node=target, role=role, observation_code=f"OBS:{n}")
        if not same_source:
            i.update(correlation_group=f"group:{n}", dependence_block=f"block:{n}")
        o["reference"]["code"] = f"OBS:{n}"
        o["reference"]["checksum"] = digest([n, status])
        o.update(status="obtained" if status == "reference_missing" else status)
        if status == "missing":
            o["value"] = None
        policy["items"].append(i)
        material["observations"].append(o)
    policy["groups"][0]["slots"] = [[i["id"]] for i in policy["items"]]
    if shared_product:
        for item in policy["items"]:
            item["observation_code"] = "OBS:0"
        material["observations"] = material["observations"][:1]
    if alternative_slot:
        policy["groups"][0]["slots"] = [[i["id"] for i in policy["items"]]]
    policy = validate(TaskPolicyV2, policy)
    material["scope"]["policy_checksum"] = digest(policy)
    material["policy_ref"]["checksum"] = digest(policy)
    material["scope"].pop("material_checksum")
    material["scope"]["material_checksum"] = digest(material)
    reader.state = validate(
        DomainState,
        state.model_copy(
            update=dict(
                policy=policy,
                material=validate(Material, material),
                nodes=tuple(sorted(set(targets))),
            )
        ),
    )
    return reader, binding


def reply_for(statuses, directions=None):
    def reply(body):
        value = review_from_wire(body)
        for index, (status, row) in enumerate(zip(statuses, value["items"], strict=True)):
            if status == "reference_missing":
                row.update(
                    status=status,
                    judgment=dict(
                        value=None,
                        lower=None,
                        upper=None,
                        reason="Synthetic reference absent",
                        method="not_assessed",
                        missing_reason="Reference not installed",
                    ),
                )
            if directions and status == "obtained":
                e = directions[index]
                row["judgment"].update(value=e, lower=min(0.0, e), upper=max(0.6, e))
        return value

    return reply


def accepted(
    *,
    statuses=("obtained",),
    roles=None,
    targets=None,
    directions=None,
    same_source=True,
    shared_product=False,
    alternative_slot=False,
):
    reader, binding = configured(
        statuses=statuses,
        roles=roles,
        targets=targets,
        same_source=same_source,
        shared_product=shared_product,
        alternative_slot=alternative_slot,
    )
    f = setup([reply_for(statuses, directions)], reader=reader, binding=binding)
    parent = f.reader.state
    candidate = f.ports.review.call(
        parent.material.scope, deadline=f.binding.deadline, round_id="A"
    )
    sub = submission(f.reader, candidate, f.budget)
    assert f.acceptance.submit(sub, key="path-source")["status"] == "submitted_not_accepted"
    assert f.acceptance.accept(key="path-source").status == "accepted"
    reader = ResearchPathReader(acceptance=f.acceptance, parent=parent, key="path-source")
    service = PathGraphValidation(reader=reader, documents=f.documents)
    return SimpleNamespace(f=f, reader=reader, service=service, parent=parent)
