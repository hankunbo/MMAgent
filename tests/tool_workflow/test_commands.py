import dataclasses

import pytest

from packages.grounding.models import canonical, digest, validate
from packages.tool_workflow import ExecutionPlan, FormalContext
from tests.tool_admission.support import reseal
from tests.tool_workflow.support import prepared


def parameter_schema(spec):
    return dict(
        type="object", properties={"threshold": spec}, required=[], additionalProperties=False
    )


def test_typed_parameter_and_explicit_default(tmp_path):
    f = prepared(
        tmp_path,
        parameters={"threshold": 0.5},
        schema=parameter_schema(dict(type="number", minimum=0, maximum=1, default=0.5)),
        argv_tail=("--threshold", "{param:threshold}"),
    )
    plan = f.compiler.compile(f.context, f.choice)
    assert plan.document.steps[0].argv[-2:] == ("--threshold", "0.5")
    assert plan.document.steps[0].effective_parameters == {"threshold": 0.5}
    assert f.compiler.validate_plan(plan) == plan


@pytest.mark.parametrize(
    "parameters,spec",
    [
        ({"threshold": 2}, dict(type="number", minimum=0, maximum=1)),
        ({"threshold": True}, dict(type="number")),
        ({"threshold": "bad"}, dict(type="string", enum=["ok"])),
        ({"extra": 1}, dict(type="number")),
    ],
)
def test_invalid_parameters_rejected_by_actual_admission(tmp_path, parameters, spec):
    with pytest.raises(AssertionError, match="parameter_invalid"):
        prepared(
            tmp_path,
            parameters=parameters,
            schema=parameter_schema(spec),
            argv_tail=("{param:threshold}",),
        )


@pytest.mark.parametrize("value", ["-c", "--help", "a/b.py", "x;rm", "{input:x}", "x\ny"])
def test_parameter_cannot_inject_interpreter_option_or_script(tmp_path, value):
    f = prepared(
        tmp_path,
        parameters={"threshold": value},
        schema=parameter_schema(dict(type="string", enum=[value])),
        argv_tail=("{param:threshold}",),
    )
    with pytest.raises(ValueError, match="unsafe_string_parameter"):
        f.compiler.compile(f.context, f.choice)


@pytest.mark.parametrize(
    "tail,reason",
    [
        (("--value={param:threshold}",), "embedded_placeholder"),
        (("{input:unknown}",), "unknown_argv_port"),
        (("{param:threshold}",), "effective_parameter_binding_missing"),
        (("{input:disc_mask}",), "duplicate_token_binding|unknown_argv_port"),
    ],
)
def test_invalid_argv_templates(tmp_path, tail, reason):
    f = prepared(tmp_path, argv_tail=tail)
    with pytest.raises(ValueError, match=reason):
        f.compiler.compile(f.context, f.choice)


def test_default_is_never_filled_under_old_admission(tmp_path):
    f = prepared(tmp_path, schema=parameter_schema(dict(type="number", default=0.5)))
    with pytest.raises(ValueError, match="effective_default_binding_missing"):
        f.compiler.compile(f.context, f.choice)
    assert all(not i.parameters for i in f.context.selection.invocations)


@pytest.mark.parametrize(
    "change", ["argv", "resource", "entrypoint", "interpreter", "profile", "config", "schema"]
)
def test_allowlist_exact_bindings(tmp_path, change):
    f = prepared(tmp_path)
    for key, a in f.reader.values.items():
        changes = {
            "argv": {"argv": (*a.argv, "--extra")},
            "resource": {
                "resources": [a.resources[0].model_copy(update={"uri": "file:///tmp/evil.py"})]
            },
            "entrypoint": {"entrypoint_checksum": digest("other")},
            "interpreter": {"executable_uri": "file:///tmp/another-interpreter"},
            "profile": {"profile_checksum": digest("other")},
            "config": {"compiler_config_checksum": digest("other")},
            "schema": {"parameter_schema_checksum": digest("other")},
        }[change]
        f.reader.values[key] = reseal(a, **changes)
    with pytest.raises(ValueError):
        f.compiler.compile(f.context, f.choice)


@pytest.mark.parametrize("change", ["allowlist", "schema_version", "policy"])
def test_exact_version_changes_invalidate_existing_plan(tmp_path, change):
    f = prepared(tmp_path)
    plan = f.compiler.compile(f.context, f.choice)
    for key, a in f.reader.values.items():
        if change == "allowlist":
            a = reseal(a, version="2.0.0")
        elif change == "schema_version":
            a = reseal(a, parameter_schema_version="2.0.0")
        else:
            f.compiler.config = reseal(f.config, version="2.0.0")
            a = reseal(a, compiler_config_checksum=f.compiler.config.checksum)
        f.reader.values[key] = a
    new = f.compiler.compile(f.context, f.choice)
    assert new.checksum != plan.checksum
    with pytest.raises(ValueError, match="stale_or_modified_plan"):
        f.compiler.validate_plan(plan)


def test_undeclared_approval_and_attempt_path_forbidden(tmp_path):
    f = prepared(tmp_path)
    plan = f.compiler.compile(f.context, f.choice)
    for key, value in [("approved", True), ("attempt_directory", "/tmp/random")]:
        data = plan.dry_run()
        data[key] = value
        data.pop("checksum")
        data["checksum"] = digest(data)
        forged = dataclasses.replace(plan, _document_json=canonical(data).decode())
        with pytest.raises(ValueError):
            f.compiler.validate_plan(forged)


@pytest.mark.parametrize("change", ["type", "unit", "laterality", "scope"])
def test_forged_input_semantics_not_accepted_with_recomputed_checksum(tmp_path, change):
    f = prepared(tmp_path)
    plan = f.compiler.compile(f.context, f.choice)
    data = plan.dry_run()
    constraint = data["steps"][0]["inputs"][0]["constraint"]
    if change == "type":
        constraint["artifact_ref"]["version"] = "9.0.0"
    elif change == "unit":
        constraint["unit"] = dict(state="known", code="CONCEPT:UNIT_MM")
    elif change == "laterality":
        constraint["laterality"] = "LEFT"
    else:
        constraint["scope_terms"] = ["CONCEPT:OTHER"]
    data.pop("checksum")
    data["checksum"] = digest(data)
    with pytest.raises(ValueError):
        f.compiler.validate_plan(ExecutionPlan(canonical(data).decode(), plan._context_json))


def test_output_target_old_bug_cannot_enter_compiler(tmp_path):
    from tests.tool_admission.test_output_targets import legacy_wrong_result

    f = prepared(tmp_path)
    selection, forged = legacy_wrong_result(f.f)
    context = validate(
        FormalContext,
        dict(request=f.f.q, query_result=f.f.result, selection=selection, admission=forged),
    )
    with pytest.raises(ValueError, match="stale_or_forged_admission_result"):
        f.compiler.compile(context, f.choice)


def test_output_filename_collision_rejected_at_controlled_reader(tmp_path):
    f = prepared(tmp_path)
    for key, a in f.reader.values.items():
        first = a.outputs[0]
        second = first.model_copy(
            update={"port": first.port.model_copy(update={"port_name": "other_output"})}
        )
        f.reader.values[key] = a.model_copy(update={"outputs": (first, second)})
    with pytest.raises(ValueError, match="output_path_collision"):
        f.compiler.compile(f.context, f.choice)


@pytest.mark.parametrize(
    "filename", ["../input.png", "/tmp/output.png", "a/b.png", ".", "a\x00.png"]
)
def test_output_escape_cannot_create_any_path(tmp_path, filename):
    f = prepared(tmp_path)
    for key, a in f.reader.values.items():
        f.reader.values[key] = a.model_copy(
            update={"outputs": (a.outputs[0].model_copy(update={"filename": filename}),)}
        )
    with pytest.raises(ValueError):
        f.compiler.compile(f.context, f.choice)


@pytest.mark.parametrize(
    "spec,value,expected",
    [
        (dict(type="boolean"), True, ("true",)),
        (dict(type="string", enum=["safe"]), "safe", ("safe",)),
        (
            dict(type="array", items=dict(type="integer"), minItems=1, maxItems=3),
            [3, 1],
            ("3", "1"),
        ),
    ],
)
def test_typed_parameter_order_preserved(tmp_path, spec, value, expected):
    f = prepared(
        tmp_path,
        parameters={"threshold": value},
        schema=parameter_schema(spec),
        argv_tail=("{param:threshold}",),
    )
    plan = f.compiler.compile(f.context, f.choice)
    assert plan.document.steps[0].argv[-len(expected) :] == expected


def test_invalid_declared_default_rejected_without_filling(tmp_path):
    f = prepared(
        tmp_path,
        parameters={"threshold": 0.5},
        schema=parameter_schema(dict(type="number", minimum=0, maximum=1, default=9)),
        argv_tail=("{param:threshold}",),
    )
    with pytest.raises(ValueError, match="parameter_invalid"):
        f.compiler.compile(f.context, f.choice)


def test_native_output_multiplicity_is_explicitly_unsupported(tmp_path):
    from packages.tool_workflow.compiler import check_command

    f = prepared(tmp_path)
    node = f.context.admission.retained[0]
    manifest = f.f.owners.read_manifest(
        f.context.admission.catalog_ref, node.candidate.tool_ref, node.candidate.implementation_ref
    )
    profile = f.f.owners.profiles[manifest.checksum]
    allowlist = f.reader.values[manifest.checksum]
    manifest = reseal(
        manifest,
        ports=[
            p.model_copy(update={"cardinality_min": 2, "cardinality_max": 2})
            if p.direction == "output"
            else p
            for p in manifest.ports
        ],
    )
    allowlist = reseal(allowlist, manifest_checksum=manifest.checksum)
    with pytest.raises(ValueError, match="output_cardinality_unsupported"):
        check_command(manifest, profile, allowlist, f.config, f.f.catalog, node.candidate)
