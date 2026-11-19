"""Dependency and executable policy checks, in addition to guarded PG integration tests."""

import ast
import json
from pathlib import Path

from packages.deterministic_pipeline.toolchain import choose_workflow
from tests.deterministic_pipeline.test_reporting import facts as facts
from tests.tool_workflow.support import prepared


def test_fixed_choice_uses_actual_admission_and_compiler(tmp_path):
    p = prepared(tmp_path)
    root = p.context.admission.retained_roots[0]
    choice = choose_workflow(p.context.admission, root)
    plan = p.compiler.compile(p.context, choice)
    assert plan.document.selection.roots == (root,)
    assert plan.document.config.network == "deny"


def test_production_dependencies_have_no_llm_expert_network_or_evaluator():
    paths = list(Path("packages/deterministic_pipeline").glob("*.py")) + [
        Path("apps/deterministic_case_pipeline.py")
    ]
    denied = {
        "openai",
        "anthropic",
        "requests",
        "httpx",
        "socket",
        "urllib",
        "subprocess",
        "agents",
    }
    for path in paths:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not {n.name.split(".")[0] for n in node.names} & denied
            if isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] not in denied
                assert not {a.name for a in node.names} & {
                    "RestrictedEvaluatorReader",
                    "PreparedSnapshot",
                    "PreparedSample",
                }
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {"eval", "exec", "__import__"}


def test_fixed_choice_blocks_missing_and_input(tmp_path):
    import pytest

    from packages.deterministic_pipeline.models import PipelineError

    p = prepared(tmp_path)
    admission = p.context.admission
    root = admission.retained_roots[0]
    node = next(n for n in admission.retained if n.identity == root)
    required = next(port for port in node.inputs if port.cardinality_min > 0)
    missing = required.model_copy(update={"artifacts": (), "producer_refs": ()})
    broken = node.model_copy(
        update={"inputs": tuple(missing if port == required else port for port in node.inputs)}
    )
    # Defense at selection as well as 06's upstream AND pruning; no compiler/runner invoked.
    incomplete = admission.model_copy(
        update={"retained": tuple(broken if n.identity == root else n for n in admission.retained)}
    )
    with pytest.raises(PipelineError, match="and_input_blocked"):
        choose_workflow(incomplete, root)


def test_shadow_and_unreviewed_rules_cannot_be_decisive(facts):
    from types import SimpleNamespace as N

    from packages.deterministic_pipeline.tool_evidence import inspect_tool_evidence
    from packages.deterministic_pipeline.validation import DeterministicValidator
    from packages.grounding.models import digest, seal, validate
    from packages.tool_admission.models import AdmissionResult, AdmissionSelection, ResearchRequest
    from packages.tool_execution.models import StepResult
    from packages.tool_workflow import ExecutionPlan, ResearchContext
    from packages.tool_workflow.models import PlanBundle, PlanDocument

    validator = DeterministicValidator(None, None, None)
    # Immutable historical synthetic facts only; no runner or historical object read.
    step, plan = facts.step, facts.plan
    admission = N(pack=N(definition=N(synthetic=True)))
    rule = N(synthetic=True, expression_code="RULE:SYNTHETIC_ASSERTION")
    assert validator.eligible(N(synthetic=True), admission, rule, step, plan) == (
        True,
        "synthetic_assertion_only",
    )
    for purpose in ("shadow", "debug"):
        bundle = json.loads(plan.to_bundle_json())
        context = bundle["context"]
        formal = PlanBundle.model_validate_json(plan.to_bundle_json()).context
        request = formal.request
        retained = formal.admission.retained[0]
        research = seal(
            ResearchRequest,
            code="RESEARCH:SYNTHETIC:09E_R01",
            version="1.0.0",
            purpose=purpose,
            compatibility_policy=request.lock.policy,
            lock=dict(run_id=plan.document.run_id, catalog_ref=plan.document.catalog_ref),
            **{
                k: getattr(request, k)
                for k in (
                    "scope",
                    "window_start",
                    "window_end",
                    "snapshot_at",
                    "satisfied_terms",
                    "artifacts",
                )
            },
            candidate=retained.candidate,
            inputs=[
                dict(port=p.port, asset_codes=[a.asset_code for a in p.artifacts])
                for p in retained.inputs
            ],
            invocation=formal.selection.invocations[0],
        )
        raw_admission = context["admission"]
        raw_admission.pop("checksum")
        raw_admission.update(
            purpose=purpose,
            request_checksum=digest(research),
            selection_checksum=digest(
                AdmissionSelection(
                    candidate_refs=(digest(research.candidate),), invocations=(research.invocation,)
                )
            ),
            query_checksum=None,
            grounding_status=None,
            upstream_findings=[],
            product_demands=[],
            capability_gaps=[],
        )
        research_context = ResearchContext(
            request=research, admission=seal(AdmissionResult, **raw_admission)
        )
        document = bundle["plan"]
        document.pop("checksum")
        document.update(
            purpose=purpose,
            context_checksum=digest(research_context),
            admission_checksum=research_context.admission.checksum,
        )
        document = seal(PlanDocument, **document)
        research_plan = ExecutionPlan.from_bundle_json(
            json.dumps(
                dict(
                    plan=document.model_dump(mode="json"),
                    context=research_context.model_dump(mode="json"),
                )
            )
        )
        research_step = validate(
            StepResult, step.model_copy(update={"plan_checksum": document.checksum})
        )
        check = inspect_tool_evidence(research_step, research_plan)
        assert check.technical_pass and check.purpose == purpose
        assert not check.evidence_candidate
        assert validator.eligible(
            N(synthetic=True), admission, rule, research_step, research_plan
        ) == (False, "nondecisive_research_observation")
    assert validator.eligible(N(synthetic=False), admission, rule, step, plan) == (
        False,
        "nondecisive_research_observation",
    )
    assert validator.eligible(
        N(synthetic=True),
        admission,
        N(synthetic=True, expression_code="RULE:UNREVIEWED"),
        step,
        plan,
    ) == (False, "reviewed_interpreter_missing")


def test_forbidden_conclusions_are_not_report_choices():
    import pytest

    from packages.disease_packs import DiseasePackCatalog
    from packages.disease_packs.catalog import validate_conclusion

    catalog = DiseasePackCatalog.from_directory(Path("configs/disease_packs"))
    from packages.deterministic_pipeline.models import PipelineConfig

    cfg = PipelineConfig.model_validate_json(
        Path("configs/deterministic_pipeline/fairvlmed_gap.json").read_bytes()
    )
    for conclusion in ("definitive_poag_diagnosis", "treatment_order", "POAG", "CDR", "GCC"):
        with pytest.raises(ValueError):
            validate_conclusion(catalog.pack(cfg.pack_ref), conclusion)
