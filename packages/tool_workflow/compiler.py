"""Controlled read/revalidation gateway plus a pure deterministic DAG compiler."""

import json
import re
from urllib.parse import unquote, urlsplit

from packages.grounding.compatibility import constraint_compatibility
from packages.grounding.models import CatalogSnapshot, canonical, digest, seal, validate
from packages.grounding.query import normalized_request
from packages.integrations.medgraph.models import parse_deployment
from packages.tool_admission.method_metadata import metadata_requirements
from packages.tool_admission.models import DeploymentProfile, InputRecord
from packages.tool_admission.parameters import validate_parameters

from .models import (
    CommandAllowlist,
    CompileError,
    ExecutionPlan,
    FormalContext,
    PlanDocument,
    PlannedStep,
    ResearchContext,
    WorkflowChoice,
    compiler_config,
)

PLACEHOLDER = re.compile(r"\{(input|output|param):([a-z][a-z0-9_]{0,63})\}")


def local_path(uri):
    """Lexical deployment validation only; filesystem/TOCTOU belongs to 06B/06D."""
    parsed = urlsplit(uri)
    path = unquote(parsed.path, errors="strict")
    if (
        parsed.scheme != "file"
        or parsed.netloc
        or parsed.query
        or parsed.fragment
        or not path.startswith("/")
        or any(p in ("", ".", "..") for p in path.split("/")[1:])
        or any(ord(c) < 32 for c in path)
        or "{" in path
        or "}" in path
    ):
        raise CompileError("unsafe_deployment_uri")
    return path


def normalized_choice(choice):
    choice = validate(WorkflowChoice, choice)
    return validate(
        WorkflowChoice,
        {
            **choice.model_dump(mode="json"),
            "roots": sorted(choice.roots),
            "nodes": [
                dict(
                    candidate_ref=n.candidate_ref,
                    inputs=[
                        dict(port=i.port, sources=sorted(i.sources, key=canonical))
                        for i in sorted(n.inputs, key=lambda i: canonical(i.port))
                    ],
                )
                for n in sorted(choice.nodes, key=lambda n: n.candidate_ref)
            ],
        },
    )


def topology(admission, choice, max_steps):
    """Choose individual sources before counting cardinality or traversing dependencies."""
    nodes = {n.identity: n for n in admission.retained}
    chosen = {n.candidate_ref: n for n in choice.nodes}
    if not choice.roots:
        raise CompileError("selection_required")
    if not set(choice.roots) <= set(admission.retained_roots):
        raise CompileError("root_outside_admission")
    if not chosen.keys() <= nodes.keys():
        raise CompileError("node_outside_admission")
    active, visited, dependencies = set(), set(), {}

    def visit(identity):
        if identity in active:
            raise CompileError("dependency_cycle")
        if identity in visited:
            return
        if identity not in chosen:
            raise CompileError("dependency_selection_missing")
        if len(active) + len(visited) >= max_steps:
            raise CompileError("step_budget_exceeded")
        active.add(identity)
        n, selection = nodes[identity], chosen[identity]
        ports = {p.port: p for p in n.inputs}
        if {p.port for p in selection.inputs} != ports.keys():
            raise CompileError("complete_input_selection_required")
        dependencies[identity] = set()
        for selected in selection.inputs:
            pool = ports[selected.port]
            count = len(selected.sources)
            if not pool.cardinality_min <= count <= pool.cardinality_max:
                raise CompileError("selected_input_cardinality")
            constraint = next(i.constraint for i in n.candidate.inputs if i.port == pool.port)
            for source in selected.sources:
                if source.kind == "artifact":
                    if source.asset_code not in {a.asset_code for a in pool.artifacts}:
                        raise CompileError("artifact_outside_admission")
                else:
                    if source.candidate_ref not in pool.producer_refs:
                        raise CompileError("producer_outside_admission")
                    producer = nodes.get(source.candidate_ref)
                    if producer is None:
                        raise CompileError("dangling_producer")
                    if producer.candidate.output_port != source.port:
                        raise CompileError("producer_port_mismatch")
                    if producer.candidate.output != constraint:
                        raise CompileError("explicit_type_unit_laterality_conversion_required")
                    dependencies[identity].add(source.candidate_ref)
                    visit(source.candidate_ref)
        active.remove(identity)
        visited.add(identity)

    for root in choice.roots:
        visit(root)
    # A caller may retain unused alternatives in its selection worksheet, but they
    # cannot become execution steps. Only reachable selections enter plan semantics.
    remaining, order = set(visited), []
    while remaining:
        ready = sorted(i for i in remaining if not dependencies[i] & remaining)
        if not ready:
            raise CompileError("dependency_cycle")
        order.extend(ready)
        remaining.difference_update(ready)
    implementations = [nodes[i].candidate.implementation_ref for i in order]
    if len(set(implementations)) != len(implementations):
        raise CompileError("repeated_implementation_nodes_unsupported")
    return order, dependencies


def check_command(manifest, profile, allowlist, config, catalog, candidate):
    if (
        allowlist.manifest_checksum != manifest.checksum
        or allowlist.profile_checksum != profile.checksum
        or allowlist.compiler_config_checksum != config.checksum
    ):
        raise CompileError("allowlist_binding_mismatch")
    if allowlist.synthetic_only and not catalog.synthetic:
        raise CompileError("synthetic_allowlist_on_real_catalog")
    native = manifest.execution_manifest
    if native is None or manifest.invocation is None or manifest.parameter_schema is None:
        raise CompileError("deployment_binding_missing")
    if (
        allowlist.executable_uri != profile.runtime.interpreter_uri
        or allowlist.executable_checksum != profile.runtime.interpreter_checksum
        or allowlist.entrypoint_uri != native.entrypoint_uri
        or allowlist.argv != manifest.invocation.argv
        or allowlist.parameter_schema_checksum != digest(manifest.parameter_schema)
        or sorted(allowlist.resources, key=canonical) != sorted(native.resources, key=canonical)
    ):
        raise CompileError("exact_command_binding_mismatch")
    executable, entrypoint = (
        local_path(allowlist.executable_uri),
        local_path(allowlist.entrypoint_uri),
    )
    if allowlist.argv[:2] != (executable, entrypoint):
        raise CompileError("fixed_executable_and_entrypoint_required")
    if not any(
        r.uri == allowlist.entrypoint_uri
        and r.sha256 is not None
        and "sha256:" + r.sha256 == allowlist.entrypoint_checksum
        and r.required
        for r in allowlist.resources
    ):
        raise CompileError("entrypoint_resource_binding_missing")
    for resource in allowlist.resources:
        local_path(resource.uri)
        if resource.sha256 is None or resource.byte_size is None:
            raise CompileError("resource_checksum_required")
    if any(
        (p.cardinality_min, p.cardinality_max) != (1, 1)
        for p in manifest.ports
        if p.direction == "output"
    ):
        raise CompileError("output_cardinality_unsupported")
    outputs = {p.port_name for p in manifest.ports if p.direction == "output"}
    if {o.port.port_name for o in allowlist.outputs} != outputs or any(
        o.port.tool_ref != candidate.tool_ref or o.port.direction != "output"
        for o in allowlist.outputs
    ):
        raise CompileError("complete_output_binding_required")
    if native.working_directory is not None:
        # An attempt-local working directory is mandatory. Do not inherit a native
        # mutable directory whose side effects are outside the plan's namespace.
        raise CompileError("native_working_directory_unsupported")


def parameter_tokens(schema, parameters, name):
    if name not in parameters:
        raise CompileError("effective_parameter_binding_missing")
    spec, value = schema["properties"][name], parameters[name]
    if spec["type"] == "array":
        values, spec = value, spec["items"]
    else:
        values = [value]
    tokens = []
    for item in values:
        if type(item) is str:
            if "enum" not in spec or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", item):
                raise CompileError("unsafe_string_parameter")
            tokens.append(item)
        elif type(item) is bool:
            tokens.append("true" if item else "false")
        elif type(item) in (int, float):
            tokens.append(json.dumps(item, allow_nan=False))
        else:
            raise CompileError("parameter_expansion_unsupported")
    return tokens


def validate_schema_defaults(schema):
    """Validate declared defaults without inserting or changing any invocation value."""
    if "default" in schema:
        validate_parameters(schema, schema["default"])
    if schema["type"] == "object":
        for child in schema["properties"].values():
            validate_schema_defaults(child)
    elif schema["type"] == "array":
        validate_schema_defaults(schema["items"])


def expand_argv(allowlist, manifest, parameters, inputs, outputs):
    validate_parameters(manifest.parameter_schema, parameters)
    validate_schema_defaults(manifest.parameter_schema)
    props = manifest.parameter_schema["properties"]
    if any("default" in spec and name not in parameters for name, spec in props.items()):
        raise CompileError("effective_default_binding_missing")
    bindings = {"input": inputs, "output": outputs}
    used = {"input": set(), "output": set(), "param": set()}
    result = []
    for token in allowlist.argv:
        match = PLACEHOLDER.fullmatch(token)
        if match is None:
            if "{" in token or "}" in token or any(ord(c) < 32 for c in token):
                raise CompileError("embedded_placeholder_or_control_character")
            result.append(token)
            continue
        kind, name = match.groups()
        if name in used[kind]:
            raise CompileError("duplicate_token_binding")
        used[kind].add(name)
        if kind == "param":
            result.extend(parameter_tokens(manifest.parameter_schema, parameters, name))
        else:
            if name not in bindings[kind]:
                raise CompileError("unknown_argv_port")
            result.extend(bindings[kind][name])
    if used != {"input": set(inputs), "output": set(outputs), "param": set(parameters)}:
        raise CompileError("complete_argv_bindings_required")
    return result


def _compile_values(context, choice, config, catalog, deployments, records):
    """Pure function over detached, revalidated owner snapshots; confers no authority."""
    admission = context.admission
    order, dependencies = topology(admission, choice, config.max_steps)
    choice = validate(
        WorkflowChoice,
        {
            **choice.model_dump(mode="json"),
            "nodes": [n for n in choice.nodes if n.candidate_ref in order],
        },
    )
    nodes = {n.identity: n for n in admission.retained}
    selections = {n.candidate_ref: n for n in choice.nodes}
    ids = {i: "step_" + i.split(":")[1] for i in order}
    invocations = (
        context.selection.invocations
        if isinstance(context, FormalContext)
        else (context.request.invocation,)
    )
    invocations = {i.implementation_ref: i for i in invocations}
    all_outputs, occupied = {}, set()
    for identity in order:
        node = nodes[identity]
        manifest, profile, allowlist = deployments[identity]
        check_command(manifest, profile, allowlist, config, catalog, node.candidate)
        tool = next(
            t for t in catalog.tools if t.implementation_ref == node.candidate.implementation_ref
        )
        specs = dict(zip(tool.output_ports, tool.outputs, strict=True))
        outputs = []
        for output in sorted(allowlist.outputs, key=lambda o: canonical(o.port)):
            if output.port not in specs:
                raise CompileError("output_port_outside_catalog")
            constraint = specs[output.port]
            c = validate(
                type(node.candidate),
                {
                    **node.candidate.model_dump(mode="json"),
                    "output_port": output.port,
                    "output": constraint,
                },
            )
            metadata = metadata_requirements(
                manifest,
                c,
                context.request.purpose if isinstance(context, ResearchContext) else None,
                tool,
            )
            reasons = constraint_compatibility(
                constraint, context.request, laterality_required=metadata.laterality_required
            )
            if reasons:
                raise CompileError("output_constraint_" + reasons[0])
            profile.check_output_targets(c, context.request.satisfied_terms)
            uri = f"plan-output://{ids[identity]}/{output.filename}"
            if uri in occupied:
                raise CompileError("output_path_collision")
            occupied.add(uri)
            outputs.append(
                dict(
                    port=output.port,
                    constraint=constraint,
                    logical_uri=uri,
                    producer_step=ids[identity],
                    qc_port=output.qc_port,
                    qc_policy_ref=output.qc_policy_ref,
                )
            )
        all_outputs[identity] = outputs
    steps = []
    for identity in order:
        n = nodes[identity]
        manifest, profile, allowlist = deployments[identity]
        invocation = invocations[n.candidate.implementation_ref]
        if digest(invocation) != n.invocation_checksum:
            raise CompileError("invocation_binding_mismatch")
        inputs, local_records = [], []
        for selected in selections[identity].inputs:
            pool = next(p for p in n.inputs if p.port == selected.port)
            constraint = next(p.constraint for p in n.candidate.inputs if p.port == selected.port)
            uris, checksums, qc = [], [], []
            for source in selected.sources:
                if source.kind == "artifact":
                    artifact = next(a for a in pool.artifacts if a.asset_code == source.asset_code)
                    record = records[source.asset_code]
                    if digest(record) != artifact.record_checksum:
                        raise CompileError("input_record_binding_mismatch")
                    m = record.metadata
                    if (m.case_code, m.patient_hash, m.encounter_code) != (
                        admission.scope.case_code,
                        admission.scope.patient_hash,
                        admission.scope.encounter_code,
                    ) or m.run_id not in (None, admission.run_id):
                        raise CompileError("input_scope_mismatch")
                    uris.append(artifact.uri)
                    checksums.append(artifact.checksum)
                    local_records.append(record)
                else:
                    output = next(
                        o for o in all_outputs[source.candidate_ref] if o["port"] == source.port
                    )
                    uris.append(output["logical_uri"])
                    qc.append(
                        f"{ids[source.candidate_ref]}:{output['port'].port_name}:technical_output_qc/1.0.0"
                    )
            inputs.append(
                dict(
                    port=selected.port,
                    constraint=constraint,
                    sources=selected.sources,
                    logical_uris=uris,
                    artifact_checksums=checksums,
                    optional_absence="not_absent" if uris else "omit_token_no_value",
                    qc_before_consume=qc,
                )
            )
        argv = expand_argv(
            allowlist,
            manifest,
            invocation.parameters,
            {i["port"].port_name: i["logical_uris"] for i in inputs},
            {o["port"].port_name: [o["logical_uri"]] for o in all_outputs[identity]},
        )
        steps.append(
            dict(
                step_id=ids[identity],
                candidate_ref=identity,
                implementation_ref=n.candidate.implementation_ref,
                tool_ref=n.candidate.tool_ref,
                dependencies=sorted(ids[d] for d in dependencies[identity]),
                inputs=inputs,
                outputs=all_outputs[identity],
                argv=argv,
                effective_parameters=invocation.parameters,
                deployment=dict(
                    manifest=manifest,
                    profile=profile,
                    allowlist=allowlist,
                    probe_checksum=n.probe_checksum,
                    resources_checksum=n.resources_checksum,
                    invocation_checksum=n.invocation_checksum,
                    input_records=sorted(local_records, key=lambda r: r.artifact.asset_code),
                ),
                timeout_seconds=min(config.timeout_seconds, manifest.invocation.timeout_seconds),
                pending_dynamic_checks=n.pending_dynamic_checks,
            )
        )
    document = seal(
        PlanDocument,
        scope=admission.scope,
        run_id=admission.run_id,
        purpose=admission.purpose,
        catalog_ref=admission.catalog_ref,
        admission_checksum=admission.checksum,
        context_checksum=digest(context),
        selection=choice,
        config=config,
        steps=[validate(PlannedStep, s) for s in steps],
        risks=(
            "limits_declared_not_enforced_until_06D",
            "QC_and_clinical_validity_not_established",
        ),
        gaps=("06D_execution_authorization_required", "06D_technical_qc_port_required"),
    )
    if len(canonical(document)) > config.max_plan_bytes:
        raise CompileError("plan_byte_limit")
    return ExecutionPlan(canonical(document).decode(), canonical(context).decode())


class ToolWorkflowCompiler:
    """Assemble with the existing controlled 06B service and deployment allowlist reader."""

    def __init__(self, admission, allowlists, config):
        self.admission, self.allowlists = admission, allowlists
        self.config = compiler_config(config)

    def compile(self, context, choice):
        return self._compile(context, choice)

    def _compile(self, context, choice, *, result_consumption=False):
        kind = context.get("kind") if isinstance(context, dict) else getattr(context, "kind", None)
        if kind not in {"formal", "research"}:
            raise CompileError("context_kind_required")
        context = validate(FormalContext if kind == "formal" else ResearchContext, context)
        if result_consumption and kind != "research":
            raise CompileError("research_consumption_only")
        choice = normalized_choice(choice)
        if isinstance(context, FormalContext):
            selection = validate(
                type(context.selection),
                {
                    "candidate_refs": sorted(context.selection.candidate_refs),
                    "invocations": sorted(
                        context.selection.invocations, key=lambda i: digest(i.implementation_ref)
                    ),
                },
            )
            context = validate(
                FormalContext,
                {
                    **context.model_dump(mode="json"),
                    "request": normalized_request(context.request),
                    "selection": selection,
                },
            )
            result = self.admission.validate_result(
                context.request, context.query_result, context.selection, context.admission
            )
        else:
            if result_consumption:
                self.admission.validate_research_consumption(context.request, context.admission)
                result = context.admission
            else:
                result = self.admission.validate_research_result(context.request, context.admission)
        if result.status != "preparation_only":
            raise CompileError("admission_not_prepared")
        catalog = validate(
            CatalogSnapshot, self.admission.catalogs.read_snapshot(result.catalog_ref)
        )
        if catalog.reference() != result.catalog_ref:
            raise CompileError("catalog_binding_mismatch")
        order, _ = topology(result, choice, self.config.max_steps)
        deployments, records = {}, {}
        retained = {n.identity: n for n in result.retained}
        selections = {n.candidate_ref: n for n in choice.nodes}
        for identity in order:
            n = retained[identity]
            manifest = parse_deployment(
                self.admission.manifests.read_manifest(
                    result.catalog_ref, n.candidate.tool_ref, n.candidate.implementation_ref
                ),
            )
            profile = validate(DeploymentProfile, self.admission.profiles.read_profile(manifest))
            if (manifest.checksum, profile.checksum) != (n.manifest_checksum, n.profile_checksum):
                raise CompileError("deployment_changed_during_compile")
            allowlist = validate(
                CommandAllowlist, self.allowlists.read_allowlist(manifest, profile, self.config)
            )
            deployments[identity] = manifest, profile, allowlist
            for port in selections[identity].inputs:
                for source in port.sources:
                    if source.kind == "artifact" and source.asset_code not in records:
                        records[source.asset_code] = validate(
                            InputRecord,
                            self.admission.inputs.read_input(
                                result.scope, result.run_id, source.asset_code
                            ),
                        )
        return _compile_values(context, choice, self.config, catalog, deployments, records)

    def validate_plan(self, plan):
        """Recompile through controlled admission; an updated seal never authorizes execution."""
        if not isinstance(plan, ExecutionPlan):
            raise CompileError("execution_plan_required")
        doc = plan.document
        current = self.compile(json.loads(plan._context_json), doc.selection)
        if current.dry_run_json() != plan.dry_run_json():
            raise CompileError("stale_or_modified_plan")
        return current

    def validate_consumed_plan(self, plan):
        """Research confirmed-result-consumption/1.0.0, never an executable plan.

        The execution plan remains exact, including its original probe. Only
        execution-probe expiry is excluded from current artifact consumption.
        Normal compile/validate_plan/dispatch still enforce that expiry.
        """
        if not isinstance(plan, ExecutionPlan):
            raise CompileError("execution_plan_required")
        current = self._compile(
            json.loads(plan._context_json), plan.document.selection, result_consumption=True
        )
        if current.dry_run_json() != plan.dry_run_json():
            raise CompileError("stale_or_modified_consumed_plan")
