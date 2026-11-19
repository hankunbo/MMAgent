"""One bounded HTTP worker per dispatch, with zero SDK or nested transport retries."""

import base64
import json
import os
import subprocess
import sys
from dataclasses import dataclass, replace

from packages.grounding.models import canonical, digest, validate

from .models import RolePayload, Usage


@dataclass(frozen=True)
class Reply:
    content: bytes = b""
    error: str | None = None
    usage: Usage = Usage()
    request_checksum: str | None = None
    api_requests: int | None = 0
    diagnostic: dict | None = None
    response_confirmed: bool | None = None
    finish_reason: str | None = None


def wire_request(
    config,
    role,
    prompt,
    payload,
    schema,
    images,
    *,
    scope_binding=None,
    wire_policy=None,
    wire_mode=None,
):
    content = [{"type": "text", "text": canonical(payload).decode()}]
    for binding, raw in images:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:"
                    + binding.media_type
                    + ";base64,"
                    + base64.b64encode(raw).decode()
                },
            }
        )
    schema = strict_schema(schema)
    if scope_binding is not None and "scope_checksum" in schema.get("properties", {}):
        # Live-only wire projection; the original output models and Fake wire stay unchanged.
        schema["properties"]["scope_checksum"]["enum"] = [scope_binding]
        bind_role_identities(schema, role.role, payload, scope_binding)
        if role.role == "review" and role.prompt_version == "review-direction-2.10.0":
            # Already owner-validated by bind_role_identities above. This repeats
            # the exact schema literals for copying, never rewrites a response.
            content.append(
                {
                    "type": "text",
                    "text": "REVIEW_RESPONSE_BINDINGS:\n"
                    + canonical(
                        {
                            key: schema["properties"][key]["enum"][0]
                            for key in ("scope_checksum", "plan_checksum", "batch_id", "round")
                        }
                    ).decode(),
                }
            )
        from .review_citations import PROMPT_VERSIONS, project

        if role.role == "review" and role.prompt_version in PROMPT_VERSIONS:
            content.append(
                {
                    "type": "text",
                    "text": project(
                        schema,
                        payload,
                        primary_status=role.prompt_version
                        in {
                            "review-batch-1.3.0",
                            "review-batch-1.4.0",
                            "review-batch-1.5.0",
                            "review-batch-1.6.0",
                            "review-batch-1.7.0",
                            "review-batch-1.8.0",
                            "review-direction-2.0.0",
                            "review-direction-2.1.0",
                            "review-direction-2.3.0",
                            "review-direction-2.4.0",
                            "review-direction-2.5.0",
                            "review-direction-2.7.0",
                            "review-direction-2.8.0",
                            "review-direction-2.9.0",
                            "review-direction-2.10.0",
                        },
                        exact_primary_status=role.prompt_version
                        in {
                            "review-direction-2.1.0",
                            "review-direction-2.3.0",
                            "review-direction-2.4.0",
                            "review-direction-2.5.0",
                            "review-direction-2.7.0",
                            "review-direction-2.8.0",
                            "review-direction-2.9.0",
                            "review-direction-2.10.0",
                        },
                        definition_gate=role.prompt_version
                        in {
                            "review-direction-2.8.0",
                            "review-direction-2.9.0",
                            "review-direction-2.10.0",
                        },
                    ),
                }
            )
    shape = {"name": role.role, "strict": True, "schema": schema}
    common = {"model": config.model, "store": False}
    if config.endpoint == "chat_completions":
        body = {
            **common,
            "stream": False,
            "max_completion_tokens": role.max_output_tokens,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": content},
            ],
            "response_format": {"type": "json_schema", "json_schema": shape},
        }
        if wire_policy is not None:
            from .wire_compatibility import apply

            return apply(body, wire_policy, wire_mode)
        return body
    converted = [{"type": "input_text", "text": content[0]["text"]}]
    converted += [{"type": "input_image", "image_url": x["image_url"]["url"]} for x in content[1:]]
    return {
        **common,
        "max_output_tokens": role.max_output_tokens,
        "instructions": prompt,
        "input": [{"role": "user", "content": converted}],
        "text": {"format": {"type": "json_schema", **shape}},
    }


def bind_role_identities(schema, role, payload, scope_binding):
    """Copy owner-qualified identities, never evidence text or a model judgement."""
    if role not in {"source", "vision", "review"}:
        return
    if isinstance(payload, dict) and payload.get("schema_version") == "review-batch-input/1.0.0":
        from packages.evidence_details.review_batches import ReviewBatchPayload

        projected = validate(ReviewBatchPayload, payload)
    elif (
        isinstance(payload, dict)
        and payload.get("schema_version") == "input-scope-vision-input/1.0.0"
    ):
        from .input_scope import ScopeVisionPayload

        projected = validate(ScopeVisionPayload, payload)
    elif (
        isinstance(payload, dict) and payload.get("schema_version") == "targeted-vision-input/1.0.0"
    ):
        from packages.evidence_acquisition.models import TargetedVisionPayload

        projected = validate(TargetedVisionPayload, payload)
    else:
        projected = validate(RolePayload, payload)
    if projected.role != role or digest(projected.material.scope) != scope_binding:
        raise ValueError("wire_material_scope")
    props = schema["properties"]
    material = projected.material
    if role == "source":
        sources = [s.reference.code for s in material.sources]
        if not sources:
            raise ValueError("wire_source_bindings_empty")
        props["target"]["enum"] = [material.goal_code]
        props["target"]["description"] = (
            "Copy material.goal_code exactly. Do not calculate or choose another identity."
        )
        props["source_code"]["enum"] = sources
    elif role == "vision":
        images = [o.image for o in material.observations if o.image is not None]
        if len(images) != 1:
            raise ValueError("wire_image_binding")
        image = images[0]
        props["image_checksum"]["enum"] = [image.reference.checksum]
        props["preprocessing_checksum"]["enum"] = [image.preprocessing.checksum]
        props["location"]["enum"] = [list(image.location)]
    else:
        props["round"]["enum"] = [projected.round]
        item_name = props["items"]["items"]["$ref"].rsplit("/", 1)[-1]
        item = schema["$defs"][item_name]
        item["properties"]["assessment_id"]["enum"] = list(projected.assessment_ids)
        props["items"]["minItems"] = props["items"]["maxItems"] = len(projected.assessment_ids)
        if hasattr(projected, "batch"):
            props["plan_checksum"]["enum"] = [projected.batch.plan_checksum]
            props["batch_id"]["enum"] = [projected.batch.batch_id]


def strict_schema(schema):
    """Wire-only normalization: optional fields remain nullable but must be present."""
    value = json.loads(json.dumps(schema))

    def visit(node):
        if isinstance(node, dict):
            node.pop("default", None)
            if node.get("type") == "object":
                node["additionalProperties"] = False
                node["required"] = list(node.get("properties", {}))
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    return value


def parse_reply(raw, endpoint, request_id=None, *, retain_metadata=False):
    try:
        data = json.loads(raw)
        u = data.get("usage") or {}
        fields = {
            "input_tokens": u.get("input_tokens", u.get("prompt_tokens")),
            "output_tokens": u.get("output_tokens", u.get("completion_tokens")),
            "total_tokens": u.get("total_tokens"),
        }
        fields = {k: v if type(v) is int and v >= 0 else None for k, v in fields.items()}
        usage = Usage(
            status="observed" if any(v is not None for v in fields.values()) else "unknown",
            **fields,
        )
        meta = dict(
            usage=usage,
            request_checksum=digest(request_id) if request_id else None,
            api_requests=1,
            response_confirmed=True,
        )
        if endpoint == "chat_completions":
            choices = data.get("choices") or []
            if not choices:
                return Reply(error="no_result", **meta)
            if len(choices) != 1:
                return Reply(error="schema", **meta)
            c = choices[0]
            m = c.get("message") or {}
            if retain_metadata:
                meta["finish_reason"] = (
                    c.get("finish_reason")
                    if c.get("finish_reason")
                    in {"stop", "length", "content_filter", "tool_calls", "function_call"}
                    else None
                )
            if m.get("refusal"):
                return Reply(error="refusal", **meta)
            if c.get("finish_reason") == "length":
                partial = m.get("content")
                return Reply(
                    error="truncated",
                    content=partial.encode()
                    if retain_metadata and isinstance(partial, str)
                    else b"",
                    **meta,
                )
            if c.get("finish_reason") != "stop" or m.get("tool_calls"):
                return Reply(error="schema", **meta)
            text = m.get("content")
        else:
            if data.get("status") == "incomplete":
                return Reply(error="truncated", **meta)
            if data.get("status") != "completed":
                return Reply(error="no_result", **meta)
            contents = [
                x
                for o in data.get("output", [])
                if o.get("type") == "message"
                for x in o.get("content", [])
            ]
            if any(x.get("type") == "refusal" for x in contents):
                return Reply(error="refusal", **meta)
            texts = [x.get("text") for x in contents if x.get("type") == "output_text"]
            text = texts[0] if len(texts) == 1 else None
        if not isinstance(text, str) or not text:
            return Reply(error="no_result", **meta)
        return Reply(content=text.encode(), **meta)
    except (ValueError, TypeError, KeyError, AttributeError):
        return Reply(error="schema", api_requests=1, response_confirmed=True)


class HTTPTransport:
    simulated = False

    def __init__(self, *, network_route=None, http_client=None):
        from .curl_client import read_client
        from .network_route import read_route

        self.network_route = read_route(network_route)
        self.http_client = read_client(http_client)
        if self.http_client is not None and self.network_route is None:
            raise ValueError("curl_explicit_route_required")

    def observation(self, reply):
        saved = getattr(self, "_last_observation", None)
        return saved[1] if saved is not None and saved[0] is reply else None

    def _observed(self, reply, value, body, operation):
        from .http_recovery import VERSION

        diagnostic = value.get("diagnostic") or {}
        status = diagnostic.get("http_status", value.get("http_status"))
        event = None
        if type(status) is int and 100 <= status <= 599:
            event = dict(
                schema_version=VERSION,
                origin="http_worker",
                operation=operation or "chat_completions",
                http_status=status,
                category=diagnostic.get("category", "success"),
                error_type=diagnostic.get("error_type"),
                error_code=diagnostic.get("error_code"),
                error_param=diagnostic.get("error_param"),
                schema_paths=diagnostic.get("schema_paths", []),
                schema_rules=diagnostic.get("schema_rules", []),
                retry_after=diagnostic.get("retry_after"),
                wire_checksum=digest(body),
                schema_checksum=digest(body.get("response_format")),
                response_model=value.get("body", {}).get("model"),
                remote_request_checksum=reply.request_checksum,
            )
        self._last_observation = (reply, event)
        return reply

    def send(
        self,
        config,
        body,
        *,
        timeout,
        max_bytes,
        operation=None,
        effective_http_timeout=None,
        timing_events=False,
    ):
        envelope = canonical(
            dict(
                config=config.model_dump(mode="json"),
                body=body,
                max_bytes=max_bytes,
                operation=operation,
                **(
                    dict(network_route=self.network_route.model_dump(mode="json"))
                    if self.network_route is not None
                    else {}
                ),
                **(dict(timing_events=True) if timing_events else {}),
                **(
                    dict(http_client=self.http_client.model_dump(mode="json"))
                    if self.http_client is not None
                    else {}
                ),
                **(
                    dict(effective_http_timeout=min(config.timeout_seconds, effective_http_timeout))
                    if effective_http_timeout is not None
                    else {}
                ),
            )
        )
        try:
            result = subprocess.run(
                [sys.executable, "-m", "packages.llm_provider.worker"]
                + (["--parent-pid", str(os.getpid())] if self.http_client is not None else []),
                input=envelope,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
            output = result.stdout
        except subprocess.TimeoutExpired as exc:
            # Worker ACK is emitted immediately at the dispatch boundary, never credentials.
            observed = 1 if (exc.stdout or b"").startswith(b"DISPATCH\n") else None
            from .transport_timing import diagnostic

            return Reply(
                error="timeout",
                api_requests=observed,
                diagnostic=diagnostic(exc.stderr, timeout_origin="outer_worker_deadline")
                if timing_events
                else None,
            )
        except Exception:
            return Reply(error="transport", api_requests=0)
        observed = 1 if output.startswith(b"DISPATCH\n") else 0
        output = output.removeprefix(b"DISPATCH\n")
        try:
            value = json.loads(output)
            if timing_events:
                from .transport_timing import diagnostic

                value["diagnostic"] = {
                    **(value.get("diagnostic") or {}),
                    **diagnostic(
                        getattr(result, "stderr", b""),
                        timeout_origin="worker_socket_timeout"
                        if value.get("error") == "timeout"
                        else None,
                    ),
                }
            if result.returncode or value.get("error"):
                safe = {
                    "authentication",
                    "timeout",
                    "rate_limit",
                    "transport",
                    "schema",
                    "secret_response",
                    "truncated",
                    "configuration",
                }
                e = value.get("error")
                return self._observed(
                    Reply(
                        error=e if e in safe else "transport",
                        api_requests=observed,
                        diagnostic=value.get("diagnostic"),
                        response_confirmed=True
                        if value.get("diagnostic", {}).get("http_status")
                        else False
                        if observed == 0
                        else None,
                    ),
                    value,
                    body,
                    operation,
                )
            if operation == "models":
                return Reply(
                    content=canonical(value["body"]), api_requests=observed, response_confirmed=True
                )
            reply = parse_reply(
                json.dumps(value["body"]).encode(),
                config.endpoint,
                value.get("request_id"),
                retain_metadata=effective_http_timeout is not None,
            )
            if timing_events:
                reply = replace(reply, diagnostic=value.get("diagnostic"))
            return self._observed(reply, value, body, operation)
        except (ValueError, KeyError, TypeError):
            return Reply(error="transport", api_requests=observed)


class DeterministicFakeTransport:
    simulated = True

    def __init__(self, replies=()):
        self.replies = list(replies)
        self.calls = []

    def send(self, config, body, *, timeout, max_bytes):
        self.calls.append(body)
        if not self.replies:
            return Reply(error="no_result")
        value = self.replies.pop(0)
        if callable(value):
            value = value(body)
        if isinstance(value, Reply):
            return value
        return Reply(content=value if isinstance(value, bytes) else canonical(value))
