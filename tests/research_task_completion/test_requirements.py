"""Original compiler/06/H configuration only; no PG or tool launch/success claim."""

from types import SimpleNamespace

import pytest

from packages.case_score_ledger.models import ScoreDefinition
from packages.case_score_ledger.service import CaseScoreService, ToolObservation
from packages.grounding.models import digest, seal
from packages.research_runtime.models import ResearchRunContext
from packages.research_runtime.task_completion import ResearchTaskCompletion
from packages.tool_execution import ToolExecutionService
from packages.tool_workflow import ResearchContext
from tests.research_task_completion.support import registered_request
from tests.tool_execution.support import prepared
from tests.tool_workflow.support import choose


def test_original_preexecution_requirements_and_profile_across_independent_registrations(tmp_path):
    ready = prepared(tmp_path / "tool")
    owner = ResearchTaskCompletion(
        SimpleNamespace(config_checksum=digest("install")), goal="GOAL:SYNTHETIC"
    )
    definitions = []
    for identity in ("one", "two"):
        case, req = registered_request(ready, identity)
        context = seal(
            ResearchRunContext, request=req, synthetic=True, scope_checksum=digest(req.scope)
        )
        admission = ready.compiler.admission.admit_research(req)
        plan = ready.compiler.compile(
            ResearchContext(request=req, admission=admission), choose(admission)
        )
        executor = ToolExecutionService(
            compiler=ready.compiler,
            catalogs=ready.f.catalogs,
            objects=ready.f.objects,
            qc=ready.qc,
            repository=SimpleNamespace(),
            policy=ready.policy.model_copy(
                update={"purpose": "shadow", "allowed_plan_checksums": (plan.checksum,)}
            ),
            workspace=ready.root / "unexecuted",
        )
        definition = ScoreDefinition(
            code="fixture.raw",
            layer="model",
            metric="logit",
            unit="logit",
            definition="Synthetic result, not disease",
            direction="higher",
        )
        output = ready.qc.policy_for(plan.document.steps[0].outputs[0]).artifact_type_code
        scores = CaseScoreService(
            runtime=SimpleNamespace(),
            executions=SimpleNamespace(),
            objects=ready.f.objects,
            observations=(
                ToolObservation(
                    key="key:tool",
                    plan_json=plan.to_bundle_json(),
                    compiler=ready.compiler,
                    outputs=((output, definition, "synthetic_png_logit/1.0.0", "value"),),
                ),
            ),
            synthetic=True,
            context_version="research-run/1.0.0",
        )
        d = owner.definition(context, plan, executor, scores, "key:tool")
        assert (
            d.case_code == case.case_code and d.requirements["outputs"] and d.requirements["scores"]
        )
        assert (
            d.execution_policy == digest(executor._policy)
            and d.score_policy == scores.policy_checksum
        )
        with pytest.raises(ValueError, match="task_completion_score_profile"):
            owner.definition(context, plan, executor, scores, "wrong-key")
        definitions.append(d)
    assert definitions[0].profile == definitions[1].profile
    assert definitions[0].context_checksum != definitions[1].context_checksum
    assert definitions[0].request_checksum != definitions[1].request_checksum
    assert definitions[0].scope_checksum != definitions[1].scope_checksum
    assert definitions[0].plan_checksum != definitions[1].plan_checksum
