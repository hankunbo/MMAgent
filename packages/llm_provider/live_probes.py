"""Nonpatient capability probes. All requests share the actual Runtime budget."""

import json
import struct
import zlib

from agents.provider_agents import StructuredRoleAgent
from packages.evidence_details.acceptance_models import DomainState, TaskPolicyV2
from packages.evidence_details.review_tasks import project_review_tasks
from packages.evidence_scoring.continuous_policy import POLICY
from packages.grounding.models import digest, validate
from packages.object_store import checksum
from packages.planner_control.models import Reference

from .models import CatalogRequestRecord, Material, Usage
from .provider import ProviderError


def red_png():
    def chunk(kind, data):
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 32, 32, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress((b"\x00" + b"\xff\x00\x00" * 32) * 32))
        + chunk(b"IEND", b"")
    )


class PublicProbeMaterials:
    def __init__(self, run_id, config):
        def ref(role, code, sha=None):
            return Reference(role=role, code=code, checksum=sha or digest(code), version="1.0.0")

        source = dict(
            reference=ref("source", "SOURCE:PUBLIC_ARITHMETIC").model_dump(mode="json"),
            locator="Public arithmetic statement",
            purpose="research_material",
            state="research_only",
            release=None,
            text="Two plus two equals four.",
        )
        definition = dict(
            examination="Compare the arithmetic observation with the exact source.",
            applicability_conditions="Public synthetic nonpatient arithmetic only.",
            interpretation="The sum 2 + 2 is 4; matching values support this arithmetic claim.",
            interpretation_status="installed",
            source_refs=[source["reference"]],
        )
        definition["reference"] = ref(
            "definition", "POLICY:PROBE_DEFINITION", digest(definition)
        ).model_dump(mode="json")
        self.raw = red_png()
        image = dict(
            reference=ref("image", "ASSET:PUBLIC_RED_SQUARE", checksum(self.raw)).model_dump(
                mode="json"
            ),
            media_type="image/png",
            width=32,
            height=32,
            coordinate_system="pixel_xy_top_left",
            preprocessing=ref("preprocessing", "POLICY:PUBLIC_PNG").model_dump(mode="json"),
            location=[0, 0, 32, 32],
        )
        observations = [
            dict(
                reference=ref("observation", "OBS:PUBLIC_SUM").model_dump(mode="json"),
                status="obtained",
                method="arithmetic",
                unit="unitless",
                value="2 + 2 = 4",
                image=None,
            ),
            dict(
                reference=ref("observation", "OBS:PUBLIC_IMAGE").model_dump(mode="json"),
                status="unknown",
                method="synthetic_shape",
                unit="pixel",
                value=None,
                image=image,
            ),
        ]
        policy = validate(
            TaskPolicyV2,
            dict(
                version="public-probe-1.0.0",
                scale=ref("scale", "POLICY:PROBE_SCALE"),
                aggregation_checksum=digest(POLICY),
                items=[
                    dict(
                        id="probe:sum",
                        node="GOAL:PUBLIC_SUM",
                        kind="measurement",
                        role="diagnostic",
                        observation_code="OBS:PUBLIC_SUM",
                        method="arithmetic",
                        unit="unitless",
                        laterality="UNKNOWN",
                        source_codes=["SOURCE:PUBLIC_ARITHMETIC"],
                        applicable=True,
                        correlation_group="public_sum",
                        dependence_block="public_sum",
                        dependency_paths=[["INPUT:PUBLIC_SUM"]],
                        definition=definition,
                    )
                ],
                groups=[
                    dict(
                        id="public_sum",
                        slots=[["probe:sum"]],
                        importance=dict(slo_image_research=1.0, comprehensive_research=1.0),
                    )
                ],
            ),
        )
        material = dict(
            scope=dict(
                case_code="CASE:PUBLIC_PROBE",
                run_id=run_id,
                graph=dict(
                    **ref("graph", "ASSET:PUBLIC_PROBE").model_dump(mode="json"),
                    kind="research_artifact",
                    parent=None,
                ),
                visible_checksum=digest("public_only"),
                input_checksum=digest("public_only"),
                policy_checksum=digest(policy),
            ),
            goal_code="GOAL:PUBLIC_SUM",
            goal_summary=(
                "Nonpatient capability probe: sum arithmetic; for image describe its "
                "actual dominant color."
            ),
            visibility="approved_visible_projection/1.0.0",
            sources=[source],
            observations=observations,
            policy_ref=ref("policy", "POLICY:PUBLIC_PROBE", digest(policy)).model_dump(mode="json"),
            missing=[],
            memory=[],
        )
        material["scope"]["material_checksum"] = digest(material)
        self.state = DomainState(
            material=validate(Material, material),
            policy=policy,
            laterality="UNKNOWN",
            resources=(ref("input", "INPUT:PUBLIC_SUM"),),
            nodes=("GOAL:PUBLIC_SUM",),
            config_checksum=digest(config),
            prompt_checksum=next(r.prompt_checksum for r in config.roles if r.role == "review"),
        )

    def read(self, scope):
        if scope != self.state.material.scope:
            raise ValueError("public_probe_scope")
        return self.state.material

    def image(self, binding):
        if binding != self.state.material.observations[1].image:
            raise ValueError("public_probe_image")
        return self.raw

    def review_template(self, scope):
        self.read(scope)
        return project_review_tasks(self.state)


def probe_catalog(*, provider, budget, materials, documents):
    saved = documents.read("probe:models:result")
    if saved:
        if saved[0]["status"] != "passed":
            raise ProviderError("configuration")
        return saved[0]
    with documents.session():
        if documents.read("probe:models:claim"):
            raise ValueError("catalog_unknown_no_repeat")
        budget.before_reserve(digest(provider.config))
        documents.write("probe:models:claim", (dict(config_checksum=digest(provider.config)),))
    start = budget.clock()
    scope = materials.state.material.scope
    request_id, attempt, deadline = budget.reserve(scope, budget.deadline, digest(provider.config))
    dispatch_error = None
    try:
        budget.check_control()
    except Exception as error:
        from .transport import Reply

        dispatch_error = error
        reply = Reply(error="configuration", api_requests=0, response_confirmed=False)
    else:
        reply = provider.transport.send(
            provider.config.transport,
            {},
            operation="models",
            timeout=min(120, (deadline - start).total_seconds()),
            max_bytes=2_000_000,
        )
    value = json.loads(reply.content) if reply.content else {}
    error = reply.error or (None if value.get("model_listed") is True else "configuration")
    record = CatalogRequestRecord(
        request_id=request_id,
        invocation_id=request_id,
        retry_index=0,
        role="source",
        scope=scope,
        provider=provider.config.transport.provider,
        model=provider.config.transport.model,
        config_checksum=digest(provider.config),
        prompt_version="models-get-1.0.0",
        prompt_checksum=digest("GET /v1/models"),
        input_checksum=digest({}),
        output_checksum=digest(value),
        remote_request_checksum=None,
        attempt=attempt,
        simulated=False,
        api_requests=reply.api_requests,
        usage=Usage(),
        latency_seconds=max(0.0, (budget.clock() - start).total_seconds()),
        error=error,
        deadline=deadline,
    )
    budget.record(record, response_confirmed=reply.response_confirmed)
    result = dict(
        status="passed" if error is None else "failed",
        result=value,
        request=record.model_dump(mode="json"),
        diagnostic=reply.diagnostic,
        catalog_is_not_inference_proof=True,
    )
    with documents.session():
        documents.write("probe:models:result", (result,))
    if dispatch_error is not None:
        raise dispatch_error
    if error:
        raise ProviderError(error)
    return result


def run_probes(*, provider, budget, documents):
    materials = PublicProbeMaterials(budget.run_id, provider.config)
    catalog = probe_catalog(
        provider=provider, budget=budget, materials=materials, documents=documents
    )
    results = dict(models=catalog)
    for name, role in (("text", "source"), ("image", "vision"), ("strict_schema", "review")):
        saved = documents.read("probe:" + name + ":result")
        if saved:
            results[name] = saved[0]
            if saved[0]["status"] != "passed":
                raise ProviderError(saved[0].get("error") or "scope")
            continue
        with documents.session():
            if documents.read("probe:" + name + ":claim"):
                raise ValueError("probe_unknown_no_repeat")
            budget.before_reserve(digest(provider.config))
            documents.write(
                "probe:" + name + ":claim", (dict(config_checksum=digest(provider.config)),)
            )
        try:
            candidate = StructuredRoleAgent(
                role, provider=provider, budget=budget, materials=materials
            ).call(
                materials.state.material.scope,
                deadline=budget.deadline,
                round_id="A" if role == "review" else None,
            )
        except ProviderError as error:
            records = [
                r
                for r in budget.records()
                if r.role == role
                and r.scope == materials.state.material.scope
                and r.endpoint == "chat_completions"
            ]
            failure = dict(
                status="failed",
                error=error.code,
                request_id=records[-1].request_id if records else None,
                candidate_accepted=False,
            )
            with documents.session():
                documents.write("probe:" + name + ":result", (failure,))
            raise
        if role == "source":
            valid = candidate.quote == "Two plus two equals four."
        elif role == "vision":
            valid = candidate.status == "observed" and any(
                t in candidate.finding.lower() for t in ("red", "红")
            )
        else:
            from packages.evidence_scoring.accepted import build_accepted

            build_accepted(materials.state, candidate)
            valid = candidate.items[0].status == "assessed"
        record = budget.records()[-1]
        result = dict(
            status="passed" if valid else "business_invalid",
            candidate=candidate.model_dump(mode="json"),
            request_id=record.request_id,
            usage=record.usage.model_dump(mode="json"),
            image_bytes=len(materials.raw) if role == "vision" else 0,
            max_completion_tokens=True,
            response_format="json_schema",
            stream=False,
        )
        with documents.session():
            documents.write("probe:" + name + ":result", (result,))
        results[name] = result
        if not valid:
            raise ProviderError("scope")
    return results
