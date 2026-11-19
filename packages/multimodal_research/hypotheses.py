"""Versioned research question projection; no hidden labels or synthetic VF."""

from apps.vectorengine_case_materials import ref
from packages.evidence_details.acceptance_models import TaskPolicyV6
from packages.grounding.models import digest, validate

REGISTRY = "configs/research/multimodal_tasks_four_hypotheses_v1.json"
LABELS = (
    ("non_glaucoma", "无青光眼", "non_glaucoma"),
    ("early_glaucoma", "早期青光眼", "early_glaucoma"),
    ("intermediate_glaucoma", "中期青光眼", "intermediate_advanced_glaucoma"),
    ("advanced_glaucoma", "晚期青光眼", "intermediate_advanced_glaucoma"),
)


def enabled(task):
    if task.code != "TASK:GLAUCOMA_FOUR_HYPOTHESES":
        return False
    from packages.dataset_adapters.multimodal.tasks import load_intent

    if load_intent("glaucoma_four_hypotheses", REGISTRY)[0] != task:
        raise ValueError("four_hypothesis_task_binding")
    return True


def expand_requirements(rows, medical, task):
    """Split the medical questions, not the input bytes or independent acquisition."""
    from .single_proposition import selected

    if not enabled(task) and selected(task) is None:
        return rows
    from .material import definition

    questions = {
        "fundus": (
            (
                "cup_rim",
                "杯盘形态与盘沿宽度",
                "描述杯盘和盘沿形态与可见范围，不能用固定杯盘比确诊或分期。",
            ),
            ("notch", "局灶盘沿切迹", "描述能看到的局灶切迹及其位置；不可见区域不是正常或阴性。"),
            (
                "rnfl",
                "眼底可见RNFL缺损",
                "只描述照片中实际可见的神经纤维层缺损；不推断未显示的OCT厚度。",
            ),
            (
                "hemorrhage",
                "视盘及盘旁出血",
                "分开已检查与未检查区域，出血只是相关线索而非分级阈值。",
            ),
        ),
        "oct": (
            (
                "inner",
                "OCT内层结构与局灶改变",
                "描述已展示切片内层结构，保留采样比例和未见区域；无分割及尺度不报微米厚度。",
            ),
            (
                "alternative",
                "OCT其他可见改变及替代解释",
                "描述可见的其他结构改变及能否解释观察；不凭单张或拼图确诊其他疾病或排除青光眼。",
            ),
        ),
    }
    result = []
    for row in rows:
        parts = questions.get(row["category"])
        if parts is None:
            result.append(row)
            continue
        for suffix, title, interpretation in parts:
            if tuple(row.get("input_modalities", ())) == ("SLO",):
                interpretation += (
                    " 本例输入为灰度SLO，不能使用颜色判据或把明暗异常直接认定为出血；"
                    "仅记录可解释外观、范围及替代解释，物理尺度和OCT厚度不可由此推定。"
                )
            identifier = row["id"] + ":" + suffix
            definition_ = row["definition"]
            body = definition_.model_dump(mode="json")
            body.update(examination=title, interpretation=interpretation)
            body["reference"] = ref(
                "definition",
                "POLICY:MM:" + digest(identifier)[7:23],
                digest({k: v for k, v in body.items() if k != "reference"}),
            )
            result.append(dict(row, id=identifier, definition=validate(type(definition_), body)))
    # A demand is not an absent result slot: it remains a gap, never a fabricated score row.
    if not any(r["category"] in {"vf", "md"} for r in result):
        result.append(
            dict(
                id="gap:functional_stage",
                component=None,
                category="vf",
                role="diagnostic",
                definition=definition(
                    "gap:functional_stage",
                    "可靠视野与功能分级依据",
                    None,
                    (medical["aao_context"],),
                ),
                input_modalities=("VF",),
                basis_kind="qualitative",
                missing_reason="no_visible_functional_staging_input",
            )
        )
    return result


def expand_policy(policy):
    """One evidence unit -> separate source-bound questions for every hypothesis."""
    raw = policy.model_dump(mode="json")
    requirements, items, bindings, units = [], [], {}, {}
    original_items = {i["id"]: i for i in raw["items"]}
    hypotheses = []
    for label, title, evaluation_label in LABELS:
        code = "GOAL:" + label.upper()
        meaning = (
            "本条只评价命题「" + title + "」的支持/反驳。e>0支持此命题，e<0反驳此命题，"
            "不是笼统的患病/损害方向。若观察仅说明存在损害而不能区分阶段，阶段特异性必须降低；"
            "无法区分时e可为0或判读缺项，不能用相同的损害评分冒充不同分期。"
            "年龄不是分期依据；照片/OCT结构与功能分期不等价。禁止虚构视野、使用隐藏标签，"
            "禁止将旧综合分切成早中晚。这里是未校准研究命题评分，不是临床分级。"
        )
        hypotheses.append(
            dict(
                code=code,
                label=label,
                title=title,
                meaning=meaning,
                evaluation_label=evaluation_label,
            )
        )
        for r in raw["requirements"]:
            key = r["id"] + ":hyp:" + label
            d = dict(r["definition"])
            d["examination"] = title + " · " + d["examination"]
            d["applicability_conditions"] = (
                "本条只评价命题「"
                + title
                + "」：正e支持该命题，负e反驳。不能区分分期时e=0或判读缺项；"
                "禁止从年龄、旧总分或未观察视野推断分期；不是临床诊断。 "
                + d["applicability_conditions"]
            )
            d["reference"] = ref(
                "definition",
                "POLICY:MMH:" + digest(key)[7:23],
                digest({k: v for k, v in d.items() if k != "reference"}),
            ).model_dump(mode="json")
            requirements.append(dict(r, id=key, definition=d))
            bindings[key], units[key] = code, r["id"]
            if r["id"] in original_items:
                items.append(
                    dict(original_items[r["id"]], id=key, node="node:" + key, definition=d)
                )
    raw.update(
        schema_version="research-task-policy/6.0.0",
        version="four-hypothesis-1.0.0",
        hypotheses=hypotheses,
        hypothesis_bindings=bindings,
        evidence_units=units,
        items=items,
        requirements=requirements,
    )
    # The evidence coverage denominator stays the number of requested checks, not x4.
    raw["groups"] = [
        dict(g, slots=[[k for k, unit in units.items() if unit in slot] for slot in g["slots"]])
        for g in raw["groups"]
    ]
    return validate(TaskPolicyV6, raw)
