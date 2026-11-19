"""One image-only development baseline per approved new case; no automatic retry."""

import base64
import importlib.util
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from apps.vectorengine_single_case import load_config, write_once
from packages.grounding.models import digest
from packages.llm_provider.transport import HTTPTransport
from packages.object_store import checksum

from .case_execution import OUTPUT_ROOT, register
from .new_cases import save

CONFIG = Path("configs/agents/vectorengine_active_acquisition_r02.json")
PROMPT = (
    "You assess a single grayscale scanning-laser ophthalmoscopy image for research. "
    "Use only this image and the provided age. Do not infer missing OCT, visual field, "
    "IOP, laterality, diagnosis labels, or prior case findings. Return one JSON object "
    "with exactly: support_score_0_100 (number 0..100, subjective uncalibrated support "
    "for glaucomatous optic neuropathy, not a calibrated disease probability), "
    "direction (support, refute, or uncertain), observations (array of visible findings), "
    "limitations (array of missing or ungradable information), reasoning (string). "
    "50 is neutral evidence, not a diagnostic cutoff. If the image cannot be assessed "
    "use null for the score and uncertain direction. Explain in Chinese. "
    "Do not use external tools or search."
)


class DirectResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    support_score_0_100: float | None = Field(ge=0, le=100, allow_inf_nan=False)
    direction: Literal["support", "refute", "uncertain"]
    observations: list[str] = Field(max_length=20)
    limitations: list[str] = Field(max_length=20)
    reasoning: str = Field(min_length=1, max_length=4000)


def body_for(binding, raw, config):
    if checksum(raw) != binding["image"]["checksum"]:
        raise ValueError("direct_image_binding")
    age = float(binding["demographics"]["age"])
    if not 0 <= age <= 120:
        raise ValueError("direct_age_invalid")
    return dict(
        model=config.transport.model,
        store=False,
        stream=False,
        max_completion_tokens=4096,
        response_format=dict(type="json_object"),
        messages=[
            dict(role="system", content=PROMPT),
            dict(
                role="user",
                content=[
                    dict(type="text", text=json.dumps(dict(modality="SLO", age_years=age))),
                    dict(
                        type="image_url",
                        image_url=dict(
                            url="data:image/png;base64," + base64.b64encode(raw).decode()
                        ),
                    ),
                ],
            ),
        ],
    )


def dispatch(body, config, directory, *, transport, now=None):
    """Durable once claim precedes any HTTP call; unknown/failed outcomes do not resend."""
    directory = Path(directory)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    request_sha = digest(body)
    final = directory / "result.json"
    if final.exists():
        result = json.loads(final.read_bytes())
        if result["request_checksum"] != request_sha:
            raise ValueError("direct_request_conflict")
        return result
    clock = now or (lambda: datetime.now(UTC))
    started = clock()
    deadline = started + timedelta(seconds=240)
    save(
        directory / "claim.json",
        dict(
            request_checksum=request_sha,
            started_at=started.isoformat(),
            deadline=deadline.isoformat(),
            max_requests=1,
            max_retries=0,
        ),
    )
    remaining = (deadline - clock()).total_seconds()
    if remaining <= 0:
        raise ValueError("direct_deadline_before_dispatch")
    reply = transport.send(
        config.transport,
        body,
        timeout=remaining,
        max_bytes=65536,
        effective_http_timeout=remaining,
    )
    # Existing transport owns secret-response filtering. Raw body is never printed.
    parsed, error = None, reply.error
    if not error:
        try:
            parsed = DirectResult.model_validate_json(reply.content).model_dump(mode="json")
            if parsed["support_score_0_100"] is None and parsed["direction"] != "uncertain":
                raise ValueError("direct_missing_score_direction")
        except ValueError:
            error = "schema"
    result = dict(
        schema_version="new-case-direct-result/1.0.0",
        request_checksum=request_sha,
        status="completed" if error is None else "failed",
        error=error,
        result=parsed,
        api_requests=reply.api_requests,
        retries=0,
        response_confirmed=reply.response_confirmed,
        usage=reply.usage.model_dump(mode="json"),
        started_at=started.isoformat(),
        ended_at=clock().isoformat(),
        elapsed_seconds=(clock() - started).total_seconds(),
        score_kind="uncalibrated_llm_self_report_not_graph_score",
        cost=None,
        clinical_acceptance=False,
        response_checksum=checksum(reply.content) if reply.content else None,
    )
    save(final, result)
    return result


def run(sample_id):
    import fcntl

    directory, binding, _, _ = register(sample_id)
    config = load_config(CONFIG)
    raw = (directory / "input.png").read_bytes()
    body = body_for(binding, raw, config)
    output = Path(OUTPUT_ROOT) / Path(sample_id).stem / "direct"
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (output / "lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_once(
            output / "protocol.json",
            dict(
                case_code=binding["case_code"],
                image_checksum=binding["image"]["checksum"],
                request_checksum=digest(body),
                config_checksum=digest(config),
                prompt=PROMPT,
                age_years=float(binding["demographics"]["age"]),
                label_access=False,
                tool_measurements_visible=False,
                max_requests=1,
                max_retries=0,
                deadline_seconds=240,
            ),
        )
        if (output / "result.json").exists():
            return dispatch(body, config, output, transport=None)
        if (output / "claim.json").exists():
            raise ValueError("direct_unknown_no_repeat")
        # Reuse the installed private launcher loader. Key never becomes a file/log/argument.
        spec = importlib.util.spec_from_file_location(
            "private_launcher", ".local/vectorengine/run.py"
        )
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        key_name = config.transport.api_key_env
        previous = os.environ.get(key_name)
        try:
            os.environ[key_name] = launcher.load_secret()
            return dispatch(body, config, output, transport=HTTPTransport())
        finally:
            if previous is None:
                os.environ.pop(key_name, None)
            else:
                os.environ[key_name] = previous
