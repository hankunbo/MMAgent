"""Public-only Review load ladder, using original projection and admission contracts.

This does not trim a patient's Material or issue research acceptance. Each rung
creates its own explicit synthetic parent; batching retains that complete parent.
"""

import copy
import json
from pathlib import Path

from apps.vectorengine_single_case import write_once
from packages.evaluation_preparation.terminal_cohort import implementation
from packages.evidence_details.acceptance_models import DomainState
from packages.evidence_details.review_batches import batch_payload, make_plan, proposal_type
from packages.grounding.models import digest, validate

from .config import prompt, read_provider_config, with_http_client
from .public_shape import PublicReviewMaterials
from .transport import wire_request

ROOT = Path(__file__).resolve().parents[2]
VERSION = "public-review-load-ladder/1.0.0"
SELECTED_VERSION = "public-review-load-ladder/1.1.0"
DIRECTIONAL_VERSION = "public-review-load-ladder/1.2.0"
MODELS = ("gpt-4o-2024-08-06", "gpt-6-sol", "gpt-6-astra")
RUNGS = (("single-small", 1, 1), ("single-expanded", 15, 1), ("triple-expanded", 15, 3))


def selected_provider(path, model):
    """Explicit public config only; never discover credentials or change the service."""
    target = (ROOT / path).resolve()
    if not target.is_relative_to(ROOT / "configs/agents") or target.suffix != ".json":
        raise ValueError("review_probe_public_provider_required")
    if target.stat().st_size > 65536:
        raise ValueError("review_probe_provider_size")
    config = read_provider_config(json.loads(target.read_bytes()))
    if (
        config.mode != "live"
        or config.transport.model != model
        or config.transport.base_url != "https://api.relayrouter.ai/v1"
        or config.transport.endpoint != "chat_completions"
        or config.transport.api_key_env != "MMAGENT_VECTORENGINE_API_KEY"
        or getattr(config, "network_route", None) is None
        or getattr(config, "review_batching", None) is None
    ):
        raise ValueError("review_probe_provider_scope")
    return config, dict(path=str(target.relative_to(ROOT)), checksum=digest(config))


def configuration(model, batch_size, *, provider_config=None):
    if model not in MODELS or batch_size not in (1, 3):
        raise ValueError("review_probe_configuration")
    if provider_config is not None:
        config, _ = selected_provider(provider_config, model)
        raw = config.model_dump(mode="json")
        raw["review_batching"]["batch_size"] = batch_size
        return read_provider_config(raw)
    raw = json.loads(
        (ROOT / "configs/agents/fairvlmed_relay_review_shape_proxy_v1.json").read_bytes()
    )
    raw["transport"]["model"] = model
    raw["review_batching"]["batch_size"] = batch_size
    return with_http_client(
        read_provider_config(raw),
        json.loads((ROOT / "configs/agents/system_curl_client_v1.json").read_bytes()),
    )


def public_state(config, identity, context, *, directional=False):
    if context not in (1, 15):
        raise ValueError("review_probe_context")
    raw = (
        PublicReviewMaterials(identity, config, task_count=context)
        .current()
        .model_dump(mode="json")
    )
    material = raw["material"]
    # Extra public context never becomes another independent medical claim.
    source = material["sources"][0]
    for index in range(1, context):
        extra = copy.deepcopy(source)
        extra["reference"]["code"] = f"SOURCE:PUBLIC_CONTEXT_{index:02d}"
        extra["locator"] = f"Public arithmetic context {index}"
        extra["reference"]["checksum"] = digest(
            {k: v for k, v in extra.items() if k != "reference"}
        )
        material["sources"].append(extra)
    if directional:
        from packages.evidence_details.acceptance_models import TaskPolicyV7
        from packages.evidence_scoring.task_policy import POLICY

        policy = raw["policy"]
        available = {
            o["reference"]["code"] for o in material["observations"] if o["status"] == "obtained"
        }
        # Keep unmet requirements in the denominator, not as obtained facts.
        policy.update(
            schema_version="research-task-policy/7.0.0",
            version="public-directional-probe-1.0.0",
            direction_policy="direction-assessment/1.0.0",
            target_code=material["goal_code"],
            target_summary=material["goal_summary"],
            aggregation_checksum=digest(POLICY),
            requirements=[
                dict(
                    id=i["id"],
                    definition=i["definition"],
                    input_modalities=["public_arithmetic_text"],
                    basis_kind="qualitative",
                )
                for i in policy["items"]
            ],
        )
        policy["items"] = [i for i in policy["items"] if i["observation_code"] in available]
        for item in policy["items"]:
            item["quality"] = dict(
                value=1.0,
                method="public_arithmetic_fixture",
                reason="Synthetic public text; not patient observation quality",
                source=raw["resources"][0],
            )
        for group in policy["groups"]:
            group["importance"] = dict(observed_input_research=1.0, task_requirement_research=1.0)
        raw["policy"] = validate(TaskPolicyV7, policy)
        material["scope"]["policy_checksum"] = digest(raw["policy"])
        material["policy_ref"]["checksum"] = digest(raw["policy"])
    material["scope"].pop("material_checksum")
    material["scope"]["material_checksum"] = digest(material)
    return validate(DomainState, raw)


def inventory(body, payload):
    """Exact serialized byte sizes, not estimated tokens, scores or independent evidence counts."""
    parts = body["messages"][1]["content"]
    return dict(
        wire_bytes=len(json.dumps(json.loads(json.dumps(body, sort_keys=True))).encode()),
        system_utf8_bytes=len(body["messages"][0]["content"].encode()),
        user_text_parts=[
            dict(index=n, utf8_bytes=len(part["text"].encode()), checksum=digest(part["text"]))
            for n, part in enumerate(parts)
        ],
        tasks=len(payload["assessment_ids"]),
        parent_observations=len(payload["material"]["observations"]),
        parent_sources=len(payload["material"]["sources"]),
        full_parent_retained=True,
    )


def build(model, nonce, rung, repetition, *, provider_config=None, directional_material=True):
    if len(nonce) != 32 or any(c not in "0123456789abcdef" for c in nonce):
        raise ValueError("review_probe_identity")
    if type(repetition) is not int or not 0 <= repetition < 3:
        raise ValueError("review_probe_repetition")
    choices = {name: (context, size) for name, context, size in RUNGS}
    if rung not in choices:
        raise ValueError("review_probe_rung")
    context, size = choices[rung]
    config = configuration(model, size, provider_config=provider_config)
    identity = f"RUN:PUBLIC_LADDER_{nonce}_{rung}_{repetition}".upper()
    state = public_state(
        config,
        identity,
        context,
        directional=directional_material and bool(getattr(config, "direction_policy", None)),
    )
    plan = make_plan(
        state, config.review_batching, key=f"public-ladder:{nonce}:{rung}:{repetition}"
    )
    payload = batch_payload(state, plan, plan.batches[0])
    role, text = prompt(config, "review")
    body = wire_request(
        config.transport,
        role,
        text,
        payload,
        proposal_type(config).model_json_schema(),
        (),
        scope_binding=digest(state.material.scope),
        wire_policy=config.wire_policy,
        wire_mode=config.wire_mode,
    )
    # Match canonical envelope -> worker json.dumps: same bytes, no new serialization policy.
    body = json.loads(json.dumps(body, sort_keys=True))
    return config, state, plan, body, inventory(body, payload)


def prepare(output, model, *, repetitions=3, provider_config=None):
    import uuid

    if type(repetitions) is not int or not 1 <= repetitions <= 3:
        raise ValueError("review_probe_repetitions")
    if model not in MODELS:
        raise ValueError("review_probe_model")
    selected = selected_provider(provider_config, model) if provider_config is not None else None
    binding = selected[1] if selected else None
    directional = bool(selected and getattr(selected[0], "direction_policy", None))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    nonce = uuid.uuid4().hex
    entries = []
    for repetition in range(repetitions):
        for rung, _, _ in RUNGS:
            config, state, plan, body, counts = build(
                model, nonce, rung, repetition, provider_config=provider_config
            )
            name = f"{repetition + 1:02d}-{rung}"
            folder = output / name
            folder.mkdir(mode=0o700)
            values = {
                "config": config.model_dump(mode="json"),
                "state": state.model_dump(mode="json"),
                "batch-plan": plan.model_dump(mode="json"),
                "wire": body,
            }
            for key, value in values.items():
                write_once(folder / (key + ".json"), value)
            entries.append(
                dict(
                    name=name,
                    rung=rung,
                    repetition=repetition,
                    inventory=counts,
                    checksums={k: digest(v) for k, v in values.items()},
                )
            )
    manifest = dict(
        schema_version=DIRECTIONAL_VERSION
        if directional
        else SELECTED_VERSION
        if binding
        else VERSION,
        **(dict(provider_binding=binding) if binding else {}),
        nonce=nonce,
        model=model,
        entries=entries,
        repetitions=repetitions,
        max_requests=len(entries),
        retries=0,
        http_seconds=90,
        outer_seconds=100,
        total_seconds=1000,
        patient_data=False,
        database_writes=0,
        clinical_qualification=False,
        stop_each_rung_on_failure=True,
        implementation=implementation(),
    )
    write_once(output / "plan.json", manifest)
    return manifest


def check(output):
    """Rebuild public inputs before reading credentials. Arbitrary saved payloads are forbidden."""
    output = Path(output)
    manifest = json.loads((output / "plan.json").read_bytes())
    if (
        manifest.get("schema_version") not in {VERSION, SELECTED_VERSION, DIRECTIONAL_VERSION}
        or manifest["implementation"] != implementation()
    ):
        raise ValueError("review_probe_version_or_implementation_changed")
    binding = None
    if manifest["schema_version"] in {SELECTED_VERSION, DIRECTIONAL_VERSION}:
        selected, binding = selected_provider(
            manifest["provider_binding"]["path"], manifest["model"]
        )
        if manifest["provider_binding"] != binding:
            raise ValueError("review_probe_provider_changed")
        if manifest["schema_version"] == DIRECTIONAL_VERSION and not getattr(
            selected, "direction_policy", None
        ):
            raise ValueError("review_probe_directional_provider_required")
    provider_config = binding["path"] if binding else None
    repeats = manifest["repetitions"]
    if type(repeats) is not int or not 1 <= repeats <= 3:
        raise ValueError("review_probe_repetitions")
    entries = []
    for repetition in range(repeats):
        for rung, _, _ in RUNGS:
            config, state, plan, body, counts = build(
                manifest["model"],
                manifest["nonce"],
                rung,
                repetition,
                provider_config=provider_config,
                directional_material=manifest["schema_version"] == DIRECTIONAL_VERSION,
            )
            name = f"{repetition + 1:02d}-{rung}"
            values = {
                "config": config.model_dump(mode="json"),
                "state": state.model_dump(mode="json"),
                "batch-plan": plan.model_dump(mode="json"),
                "wire": body,
            }
            for key, value in values.items():
                path = output / name / (key + ".json")
                if json.loads(path.read_bytes()) != value:
                    raise ValueError("review_probe_public_input_changed")
            entries.append(
                dict(
                    name=name,
                    rung=rung,
                    repetition=repetition,
                    inventory=counts,
                    checksums={k: digest(v) for k, v in values.items()},
                )
            )
    expected = dict(
        schema_version=manifest["schema_version"],
        **(dict(provider_binding=binding) if binding else {}),
        nonce=manifest["nonce"],
        model=manifest["model"],
        entries=entries,
        repetitions=repeats,
        max_requests=len(entries),
        retries=0,
        http_seconds=90,
        outer_seconds=100,
        total_seconds=1000,
        patient_data=False,
        database_writes=0,
        clinical_qualification=False,
        stop_each_rung_on_failure=True,
        implementation=manifest["implementation"],
    )
    if manifest != expected:
        raise ValueError("review_probe_plan_changed")
    return manifest
