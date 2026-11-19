"""Shared synthetic initial material for memory and unified owned-PG assembly tests."""

import json

from packages.grounding.models import digest, validate
from packages.llm_provider.live_probes import PublicProbeMaterials
from packages.llm_provider.models import Material
from packages.llm_provider.public_shape import PublicReviewMaterials
from tests.agent_provider.batch_support import batch_reply


def initial_state(binding, config, count=7):
    owner = PublicReviewMaterials(binding.run_id, config, task_count=count)
    probe = PublicProbeMaterials(binding.run_id, config)
    raw = owner.current().material.model_dump(mode="json")
    raw["observations"][0]["image"] = probe.state.material.observations[1].image.model_dump(
        mode="json"
    )
    raw["scope"].update(
        case_code=binding.case_code,
        graph=binding.graph.model_dump(mode="json"),
        input_checksum=binding.input_checksum,
        visible_checksum=binding.visible_checksum,
    )
    raw["scope"].pop("material_checksum")
    raw["scope"]["material_checksum"] = digest(raw)
    state = owner.current().model_copy(update={"material": validate(Material, raw)})
    return state, probe.raw


def source_reply(body):
    payload = json.loads(body["messages"][1]["content"][0]["text"])
    m = payload["material"]
    return dict(
        scope_checksum=digest(m["scope"]),
        source_code=m["sources"][0]["reference"]["code"],
        quote=m["sources"][0]["text"],
        target=m["goal_code"],
        relation="requires",
        missing=[],
        reason="Synthetic reference, not a patient conclusion",
    )


def vision_reply(body):
    payload = json.loads(body["messages"][1]["content"][0]["text"])
    m = payload["material"]
    (image,) = [o["image"] for o in m["observations"] if o["image"]]
    return dict(
        scope_checksum=digest(m["scope"]),
        image_checksum=image["reference"]["checksum"],
        preprocessing_checksum=image["preprocessing"]["checksum"],
        location=image["location"],
        status="observed",
        finding="Synthetic red square observation",
    )


REPLIES = [source_reply, vision_reply, batch_reply, batch_reply, batch_reply]
