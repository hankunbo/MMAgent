"""Explicit binary proposition tasks, sharing acquisition but never target meanings."""

from packages.dataset_adapters.multimodal.tasks import load_intent
from packages.grounding.models import digest, validate

from .hypotheses import LABELS

REGISTRY = "configs/research/multimodal_propositions_v1.json"
REGISTRIES = {
    "1.0.0": REGISTRY,
    "1.1.0": "configs/research/multimodal_propositions_v2.json",
}


def root_direction_spec(task):
    """Version the question, not a medical relation or a desired answer."""
    value = selected(task)
    return dict(
        schema_version="root-direction-question/1.0.0",
        task_checksum=task.checksum,
        target_code=task.target_code,
        title=value[1] if value else task.target_code,
        desired_direction=None,
        medical_relation_changed=False,
    )


def root_direction_state(state, task, spec):
    """Explicit opt-in material projection; old inputs and scores stay unchanged."""
    from packages.evidence_details.review_tasks import project_review_tasks

    if spec != root_direction_spec(task) or state.material.goal_code != task.target_code:
        raise ValueError("root_direction_question_binding")
    raw = state.model_dump(mode="json")
    raw["material"]["goal_summary"] = (
        f"待检验根命题：{spec['title']}（{task.target_code}）。"
        "e>0表示本例观察支持这个根命题，e<0表示反驳这个根命题；"
        "不是检查现象存在的分数，也不是图像正常、方法成功或回答正确的分数。"
        "先区分观察到什么与该观察怎样影响根命题，再给有来源依据的连续方向估计。"
        "一般损害与某一期别的关系须单独说明；不像某期不等于无病。"
        "允许很弱方向和跨零区间，不设最低幅度或预期符号；不机械反号。"
        "零须比较后无可辩护净方向，null须具体不可替代障碍。研究分非概率或诊断。"
    )
    if "target_summary" in raw["policy"]:
        raw["policy"]["target_summary"] = raw["material"]["goal_summary"]
        raw["material"]["scope"]["policy_checksum"] = digest(raw["policy"])
        raw["material"]["policy_ref"]["checksum"] = digest(raw["policy"])
    raw["material"]["scope"].pop("material_checksum")
    raw["material"]["scope"]["material_checksum"] = digest(raw["material"])
    result = validate(type(state), raw)
    project_review_tasks(result)
    return result


def selected(task):
    for label, title, _ in LABELS:
        if task.code == "TASK:PROPOSITION_" + label.upper():
            registry = REGISTRIES.get(task.version)
            if registry is None or task != load_intent(label, registry)[0]:
                raise ValueError("single_proposition_task_binding")
            return label, title
    return None


def meaning(task, *, directional=False):
    if directional:
        return (
            "只评阅指定命题；e为本例观察对该命题的连续方向，非概率。"
            "有依据的弱方向不因缺阈值或不能单项定期归零；0须比较后无净方向，null须具体判读障碍。"
            "不把一般损害自动当分期依据，不补不可见事实；质量及适用限制分别解释。"
        )
    value = selected(task)
    if value is None:
        return None
    return (
        "只评阅「" + value[1] + "」。e>0支持这个命题，e<0反驳；非此期不等于无病。"
        "理由须与正负号一致，不能将损害方向复制成分期方向。无法区分时e=0并说明范围。"
        "年龄不决定分期，结构不等于功能分期；不补造视野，不切分旧总分。研究分非概率或诊断。"
    )


def bind_requirements(rows, task, *, directional=False):
    instruction = meaning(task, directional=directional)
    if instruction is None:
        return rows
    result = []
    for row in rows:
        definition = row["definition"].model_dump(mode="json")
        definition["applicability_conditions"] = (
            instruction + " " + definition["applicability_conditions"]
        )
        definition["reference"]["checksum"] = digest(
            {k: v for k, v in definition.items() if k != "reference"}
        )
        result.append(dict(row, definition=validate(type(row["definition"]), definition)))
    return result


def clarification_spec(rows, task, assessment_ids, *, version="1.0.0"):
    """Bound an operator-selected semantic review to the original task/definitions.

    This adds no medical relation, threshold, fact, or desired answer. It is a
    versioned question projection, not a new Knowledge medical-definition grant.
    """
    ids = tuple(sorted(assessment_ids))
    by_id = {r["id"]: r for r in rows}
    if (
        version not in {"1.0.0", "1.1.0"}
        or not 1 <= len(ids) <= 2
        or len(set(ids)) != len(ids)
        or not set(ids) <= set(by_id)
    ):
        raise ValueError("review_clarification_targets")
    return dict(
        schema_version="proposition-review-clarification/" + version,
        task_checksum=task.checksum,
        target_code=task.target_code,
        assessment_ids=list(ids),
        original_definitions={key: digest(by_id[key]["definition"]) for key in ids},
        reason="root_proposition_and_observed_feature_direction_may_be_confused",
        desired_direction=None,
        medical_relation_changed=False,
    )


def clarify_requirements(rows, task, spec):
    versions = {
        "proposition-review-clarification/1.0.0": "1.0.0",
        "proposition-review-clarification/1.1.0": "1.1.0",
    }
    version = versions.get(spec.get("schema_version"))
    if version is None or spec != clarification_spec(
        rows, task, spec["assessment_ids"], version=version
    ):
        raise ValueError("review_clarification_binding")
    value = selected(task)
    title = value[1] if value else task.target_code
    result = []
    for row in rows:
        if row["id"] not in spec["assessment_ids"]:
            result.append(row)
            continue
        definition = row["definition"].model_dump(mode="json")
        if definition["interpretation_status"] != "installed":
            raise ValueError("review_clarification_requires_existing_interpretation")
        if version == "1.1.0":
            instruction = (
                f"评阅命题P={title}。e>0支持P，e<0支持非P；"
                "支持非P不能标为support或正e。图像正常不等于支持疾病。"
                "不预设方向或幅度，不机械反号；比较观察、来源和替代解释后说明净方向。"
                "0需比较理由，null需具体障碍。原判读："
            )
        else:
            instruction = (
                f"方向对象：{title}（{task.target_code}），不是检查现象是否存在。"
                "e>0支持根命题，e<0反驳。伪影/替代解释不自动支持疾病，也不机械反号；"
                "说明对根命题的影响。无预设方向/幅度；0需比较理由，null需具体障碍。原判读："
            )
        definition["interpretation"] = instruction + definition["interpretation"]
        definition["reference"].update(
            code="DEFINITION:POLARITY_" + digest((row["id"], spec))[7:31].upper(),
            version="1.0.0",
        )
        definition["reference"]["checksum"] = digest(
            {k: v for k, v in definition.items() if k != "reference"}
        )
        result.append(dict(row, definition=validate(type(row["definition"]), definition)))
    return result


def clarify_review_state(state, task, spec):
    """Project after source aliases, so the wire and accepted references agree."""
    from packages.evidence_details.review_tasks import project_review_tasks

    if state.material.goal_code != task.target_code:
        raise ValueError("review_clarification_goal_binding")
    rows = [dict(id=r.id, definition=r.definition) for r in state.policy.requirements]
    definitions = {
        r["id"]: r["definition"].model_dump(mode="json")
        for r in clarify_requirements(rows, task, spec)
    }
    raw = state.model_dump(mode="json")
    for name in ("items", "requirements"):
        for item in raw["policy"][name]:
            if item["id"] in spec["assessment_ids"]:
                item["definition"] = definitions[item["id"]]
    raw["material"]["scope"]["policy_checksum"] = digest(raw["policy"])
    raw["material"]["policy_ref"]["checksum"] = digest(raw["policy"])
    raw["material"]["scope"].pop("material_checksum")
    raw["material"]["scope"]["material_checksum"] = digest(raw["material"])
    result = validate(type(state), raw)
    project_review_tasks(result)
    return result
