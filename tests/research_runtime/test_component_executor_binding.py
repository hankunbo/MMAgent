"""No dispatch: bind the original executor without current-input IO or credentials."""

from types import SimpleNamespace

import pytest

from db.runtime.repository.case_pipeline import identity
from packages.grounding.models import seal
from packages.research_runtime.component_deployment import (
    ComponentServices,
    component_executor_factory,
)
from packages.research_runtime.models import ResearchRunContext
from packages.tool_workflow import ResearchContext
from tests.tool_admission.support import reseal
from tests.tool_execution.test_trusted_local import setup  # noqa: F401
from tests.tool_execution.trusted_support import prepared
from tests.tool_workflow.support import choose


def binding(f):
    from packages.grounding.models import digest

    run = identity("RUN", "research:" + f.request.code)
    request = reseal(f.request, lock=dict(run_id=run, catalog_ref=f.request.lock.catalog_ref))
    f.f.owners.requests[(request.code, request.version, request.checksum)] = request
    admitted = f.f.service.admit_research(request)
    f.plan = f.compiler.compile(
        ResearchContext(request=request, admission=admitted), choose(admitted)
    )
    context = seal(
        ResearchRunContext, request=request, synthetic=False, scope_checksum=digest(request.scope)
    )
    runtime = SimpleNamespace(
        _target=object(), read_research_run=lambda key: context if key == run else None
    )
    requests = SimpleNamespace(
        runtime=runtime,
        catalog_ref=f.plan.document.catalog_ref,
        request_key="fixture",
        request_code=lambda key: request.code,
        parent=SimpleNamespace(request=SimpleNamespace(scope=request.scope)),
        candidate=request.candidate,
        invocation=request.invocation,
        policy=request.compatibility_policy,
        scope_terms=request.satisfied_terms,
    )
    f.f.service.research = f.f.service.inputs = requests
    research = SimpleNamespace(
        runtime=runtime, compiler=f.compiler, research=requests, inputs=requests, synthetic=False
    )
    return ComponentServices(requests, f.f.service, f.compiler, research)


def factory(f, services):
    return component_executor_factory(
        services=services,
        catalogs=f.catalogs,
        objects=f.f.objects,
        qc=f.service._qc,
        policy={**f.service._policy.model_dump(mode="json"), "allowed_plan_checksums": []},
        workspace=f.service._runner.workspace,
    )


def test_binds_original_target_and_policy_without_dispatch(setup, monkeypatch):  # noqa: F811
    f = prepared(setup)
    services = binding(f)
    checked = []
    # CPU constructor test only: PG role validation is an original DB read,
    # never a patient-input read or a launch. No live database in this fixture.
    monkeypatch.setattr("db.runtime.repository.postgres.validate_runtime_principal", checked.append)
    before = f.service._policy.model_dump(mode="json")
    make = factory(f, services)
    first, second = make(f.plan), make(f.plan)
    assert first._repo._target is services.requests.runtime._target
    assert first._repo.compiler is services.compiler
    assert first._policy == second._policy
    assert first._policy.allowed_plan_checksums == (f.plan.checksum,)
    assert first._policy.model_dump(mode="json") == dict(
        before, allowed_plan_checksums=[f.plan.checksum]
    )
    assert f.service._policy.model_dump(mode="json") == before
    assert checked and all(target is services.requests.runtime._target for target in checked)


def test_wrong_runtime_rejected_before_policy_or_io():
    services = ComponentServices(
        SimpleNamespace(runtime=object()), None, None, SimpleNamespace(runtime=object())
    )
    with pytest.raises(ValueError, match="component_execution_runtime_binding"):
        component_executor_factory(
            services=services, catalogs=None, objects=None, qc=None, policy=None, workspace="unused"
        )


def test_wrong_catalog_and_non_plan_rejected(setup):  # noqa: F811
    f = prepared(setup)
    services = binding(f)
    make = factory(f, services)
    with pytest.raises(ValueError, match="component_one_step_plan_required"):
        make(object())
    services.requests.catalog_ref = None
    with pytest.raises(ValueError, match="component_execution_installation_binding"):
        make(f.plan)


@pytest.mark.parametrize(
    "change", ["compiler", "inputs", "request_key", "candidate", "purpose", "synthetic"]
)
def test_cross_owner_and_cross_request_rejected(setup, change):  # noqa: F811
    f = prepared(setup)
    services = binding(f)
    if change == "compiler":
        services.research.compiler = object()
    elif change == "inputs":
        services.admission.inputs = object()
    if change in {"compiler", "inputs"}:
        with pytest.raises(ValueError, match="component_execution_owner_binding"):
            factory(f, services)
        return
    make = factory(f, services)
    if change == "request_key":
        services.requests.request_code = lambda key: "RESEARCH:OTHER:CHILD"
        error = "component_execution_child_binding"
    else:
        services.requests.candidate = None
        error = "component_execution_request_binding"
        if change == "purpose":
            # Context/plan still carry shadow; a debug installation cannot adopt it.
            services.requests.candidate = f.request.candidate
            make = component_executor_factory(
                services=services,
                catalogs=f.catalogs,
                objects=f.f.objects,
                qc=f.service._qc,
                policy={
                    **f.service._policy.model_dump(mode="json"),
                    "allowed_plan_checksums": [],
                    "purpose": "debug",
                },
                workspace=f.service._runner.workspace,
            )
        elif change == "synthetic":
            services.requests.candidate = f.request.candidate
            services.research.synthetic = True
    with pytest.raises(ValueError, match=error):
        make(f.plan)
