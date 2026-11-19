"""Production component assembly is read-free; original owners own all decisions."""

from types import SimpleNamespace

import pytest

from packages.integrations.medgraph.resources import LocalResourcePolicy
from packages.research_runtime.component_deployment import assemble_component_services
from packages.research_runtime.component_inputs import ComponentResearchRequests
from tests.tool_workflow.support import config


def assemble(requests, **overrides):
    args = dict(
        requests=requests,
        catalogs=None,
        manifests=None,
        profiles=None,
        probes=None,
        allowlists=None,
        objects=None,
        resources=LocalResourcePolicy(allowed_roots=("/opt/data/test/MMAgent",)),
        config=config(),
        clock=lambda: None,
        synthetic=False,
    )
    args.update(overrides)
    return assemble_component_services(**args)


def test_requires_original_component_owner():
    with pytest.raises(ValueError, match="component_original_assembly_required"):
        assemble(SimpleNamespace())


@pytest.mark.parametrize("identity", [None, 0, 1, "false"])
def test_context_identity_is_not_truthiness(identity):
    request = object.__new__(ComponentResearchRequests)
    with pytest.raises(ValueError, match="component_original_assembly_required"):
        assemble(request, synthetic=identity)


def test_construction_does_not_read_current_inputs_or_register():
    # Constructor isolation only. Persistent integration uses a fully constructed
    # original owner, not this deliberately uninitialised test object.
    request = object.__new__(ComponentResearchRequests)
    request.runtime = object()
    services = assemble(request)
    assert services.research.runtime is request.runtime
    assert services.research.research is services.research.inputs is request
    assert services.research.compiler is services.compiler
    assert services.compiler.admission is services.admission
    assert services.admission.research is services.admission.inputs is request


def test_compiler_config_is_not_rewritten():
    from tests.tool_admission.support import reseal

    request = object.__new__(ComponentResearchRequests)
    request.runtime = object()
    configured = reseal(config(), max_steps=2)
    services = assemble(request, config=configured)
    assert services.compiler.config == configured
    # ComponentPlannerSteps, not a rewritten shared config, requires the actual
    # selected plan to contain exactly one step.
