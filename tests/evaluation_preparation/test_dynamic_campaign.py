from datetime import UTC, datetime, timedelta
from threading import Event
from types import SimpleNamespace

import pytest

from apps.vectorengine_live_assembly import run_case_stages
from packages.evaluation_preparation.dynamic_campaign import Campaign
from packages.evaluation_preparation.dynamic_stages import acquisition_state, review_state
from packages.grounding.models import digest, validate
from packages.llm_provider.models import ObservationInput
from packages.llm_provider.provider import ProviderError
from packages.object_store import checksum
from tests.evaluation_preparation.test_dynamic_evidence import dynamic_fixture
from tests.expert_agent.batch_case_support import vision_reply
from tests.expert_agent.test_batch_assembly import assembly


def test_code_only_continuation_preserves_scope_budget_and_owner(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    monkeypatch.chdir(tmp_path)
    c = object.__new__(Campaign)
    c.output = tmp_path / "case"
    c.output.mkdir()
    c.paths = lambda: [c.output / "vision", c.output / "review"]
    name = "packages/evaluation_preparation/dynamic_report.py"
    previous = dict(
        sample="case",
        max_new_requests=10,
        implementation={name: "old"},
        started_at="unchanged",
        deadline="unchanged",
    )
    expected = dict(sample="case", max_new_requests=10, implementation={name: "new"})
    gate = Path("docs/acceptance/13-dynamic-campaign-v1/test/verification.json")
    gate.parent.mkdir(parents=True)
    verified = dict(passed=True, implementation=expected["implementation"])
    gate.write_text(json.dumps(verified))
    proof = dict(
        schema_version="dynamic-code-read-port-handoff/1.0.0",
        original_protocol_checksum=digest(previous),
        current_implementation=expected["implementation"],
        verification_path=str(gate),
        verification_checksum=digest(verified),
        acquisition_results={},
    )
    (c.output / "review-implementation-handoff.json").write_text(json.dumps(proof))
    c.check_continuation(previous, expected)
    with pytest.raises(ValueError, match="dynamic_campaign_changed"):
        c.check_continuation(previous, dict(expected, max_new_requests=11))
    with pytest.raises(ValueError, match="dynamic_unreviewed_implementation_change"):
        c.check_continuation(previous, dict(expected, implementation={"provider.py": "new"}))
    (c.paths()[0]).mkdir()
    (c.paths()[0] / "result.json").write_text("{}")
    with pytest.raises(ValueError, match="dynamic_continuation_acquisition_inventory"):
        c.check_continuation(previous, expected)


def test_selected_role_uses_original_planner_acceptance_and_no_source_request(
    monkeypatch, tmp_path
):
    f = assembly(monkeypatch, replies=[vision_reply])
    results, roles = run_case_stages(
        planner=f.planner,
        owner=f.owner,
        ports=f.ports,
        review=f.review,
        documents=f.documents,
        binding=f.binding,
        config=f.config,
        output=tmp_path,
        signal_cancel=Event(),
        selected_roles=("vision",),
    )
    assert roles == ("vision",) and results[0]["domain_status"] == "accepted"
    assert len(f.wire.calls) == f.ports.budget.used == 1
    assert f.documents.read("live-stage:source") is None


def test_acquisition_unknown_is_not_a_final_result_slot(monkeypatch):
    import packages.evaluation_preparation.dynamic_stages as module
    from packages.llm_provider.live_probes import PublicProbeMaterials

    f = dynamic_fixture(1, empty=True)
    seed = f.owner.current()
    target = seed.policy.requirements[0].id
    monkeypatch.setattr(module, "VISUAL_IDS", (target,))
    probe = PublicProbeMaterials(seed.material.scope.run_id, f.config)
    image = next(o.image for o in probe.state.material.observations if o.image)
    asset = SimpleNamespace(
        dimensions=(image.height, image.width),
        asset_code=image.reference.code,
        checksum=checksum(probe.raw),
    )
    task = acquisition_state(seed, target, asset)
    assert task.material.observations[0].status == "unknown"
    assert task.policy.schema_version == "research-task-policy/2.0.0"
    empty = review_state(seed, {target: task.material.observations[0]})
    assert empty.policy.items == ()
    observed = task.material.observations[0].model_dump(mode="json")
    observed.update(status="obtained", value="Synthetic irregular boundary")
    observed["reference"]["checksum"] = digest(observed)
    result = review_state(seed, {target: validate(ObservationInput, observed)})
    assert len(result.policy.items) == 1
    assert result.policy.requirements[0].missing_reason is None
    assert result.policy.groups == seed.policy.groups
    assert result.policy.items[0].dependency_paths == ((seed.resources[0].code,),)
    with pytest.raises(ValueError, match="dynamic_observation_target"):
        review_state(seed, {"wrong": validate(ObservationInput, observed)})


@pytest.mark.parametrize(
    "prior,used,dispatch,allowed",
    [
        (18, 9, False, True),
        (18, 10, False, False),
        (18, 10, True, True),
        (18, 11, True, False),
        (39, 1, False, False),
        (39, 1, True, True),
    ],
)
def test_all_phase_reservations_share_case_cap(prior, used, dispatch, allowed):
    c = Campaign.__new__(Campaign)
    c.protocol = dict(
        previous=dict(reservations=prior),
        max_new_requests=10,
        deadline=(datetime.now(UTC) + timedelta(minutes=2)).isoformat(),
    )
    c.prior = lambda: dict(reservations=prior)
    c.usage = lambda: used
    if allowed:
        c.guard(dispatch=dispatch)
    else:
        with pytest.raises(ProviderError, match="budget"):
            c.guard(dispatch=dispatch)


def test_deadline_and_prior_drift_do_not_reset_budget():
    c = Campaign.__new__(Campaign)
    c.protocol = dict(
        previous=dict(reservations=18),
        max_new_requests=10,
        deadline=(datetime.now(UTC) - timedelta(seconds=1)).isoformat(),
    )
    c.prior = lambda: dict(reservations=18)
    c.usage = lambda: 0
    with pytest.raises(ProviderError, match="deadline"):
        c.guard()
    c.prior = lambda: dict(reservations=19)
    with pytest.raises(ProviderError, match="scope"):
        c.guard()


def test_new_prompt_and_exact_per_task_citations_are_installed():
    from apps.vectorengine_single_case import load_config
    from packages.evaluation_preparation.dynamic_campaign import CONFIG
    from packages.llm_provider.config import prompt
    from packages.llm_provider.review_citations import PROMPT_VERSIONS

    c = load_config(CONFIG)
    setting, text = prompt(c, "review")
    assert setting.prompt_version in PROMPT_VERSIONS
    assert "Source-driven qualitative" in text and "REVIEW_CITATION_CODES" in text


def test_optional_model_is_obtained_not_a_compulsory_slot():
    import json

    from packages.evaluation_preparation.dynamic_model import attach_model
    from tests.agent_provider.batch_support import batch_reply, generate

    f = dynamic_fixture(1, empty=True)
    state = f.owner.current()
    assert attach_model(state, None) is state
    model = dict(
        output=dict(normalized_model_scores=[0.6, 0.4]),
        source=dict(
            artifact_code="ASSET:SYNTHETIC_MODEL",
            artifact_checksum=digest("output"),
            implementation=dict(code="TOOL:SYNTHETIC", version="1.0.0", checksum=digest("model")),
        ),
        run_id="RUN:SYNTHETIC_MODEL",
    )
    f.owner.state = attach_model(state, model)
    assert len(f.owner.state.policy.requirements) == 2
    assert len(f.owner.state.policy.items) == 1

    def reply(body):
        raw = batch_reply(body)
        obs = f.owner.state.material.observations[0]
        for item in raw["items"]:
            item["judgment"].update(
                value=json.loads(obs.value)["model_direction"],
                lower=-0.4,
                upper=0,
                method="fixed_model_margin",
            )
        return raw

    from packages.llm_provider import DeterministicFakeProvider

    f.provider = DeterministicFakeProvider(f.config, [reply])
    f.agent.provider = f.provider
    sub = generate(f)
    f.acceptance.submit(sub, key="new-model")
    assert f.acceptance.accept(key="new-model").status == "accepted"
    row = f.acceptance.read(key="new-model")[2]["scores"]["reviews"][0]
    assert row["judgment"]["value"] == pytest.approx(-0.2)
    assert row["weight"] > 0


@pytest.mark.parametrize("structured", [False, True])
def test_dynamic_report_owner_readback_and_escape(tmp_path, structured):
    import json

    from agents.expert_agent.reporting import read_research_report
    from apps.vectorengine_single_case import write_once
    from packages.evaluation_preparation.dynamic_report import export
    from tests.agent_provider.batch_support import generate

    f = dynamic_fixture(1)
    state = f.owner.current()
    f.acceptance.submit(generate(f), key="export")
    assert f.acceptance.accept(key="export").status == "accepted"
    f.documents.write("dynamic-phase-input", (state.model_dump(mode="json"),))
    report = read_research_report(f.acceptance, key="export", parent=state)
    (tmp_path / "review").mkdir()
    write_once(tmp_path / "review/review-report.json", report)
    c = SimpleNamespace(
        structured_completion=structured,
        output=tmp_path,
        sample="synthetic.npz",
        target=None,
        documents=lambda _: f.documents,
        base=SimpleNamespace(
            directory=tmp_path, quality={"reason": "<script>not executable</script>"}
        ),
    )
    result = dict(
        phases=[dict(run_id=state.material.scope.run_id, accepted=True, acceptance_key="export")],
        new_requests=1,
        prior_graph_requests=0,
    )
    paths = export(c, result, directory=tmp_path / "exports")
    from pathlib import Path

    text = Path(paths["html"]).read_text()
    assert "&lt;script&gt;" in text and "<script>not executable" not in text
    assert json.loads(Path(paths["json"]).read_bytes())["acquired_count"] == 1
    saved_report = json.loads(Path(paths["json"]).read_bytes())
    assert ("completion" in saved_report) == structured
    if structured:
        assert saved_report["completion"]["all_obtained_reviewed"]
    assert export(c, result, directory=tmp_path / "exports") == paths


def test_original_expert_keeps_confirmed_batch_after_later_timeout(monkeypatch, tmp_path):
    import tests.expert_agent.test_batch_assembly as module
    from packages.evidence_details.acceptance_models import DomainState, TaskPolicyV3
    from packages.evidence_scoring.dynamic_policy import POLICY
    from packages.llm_provider.transport import Reply
    from tests.agent_provider.batch_support import batch_reply

    original = module.initial_state

    def initial(binding, config, count):
        state, image = original(binding, config, count)
        raw = state.model_dump(mode="json")
        available = {
            o.reference.code for o in state.material.observations if o.status == "obtained"
        }
        policy = raw["policy"]
        policy.update(
            schema_version="research-task-policy/3.0.0",
            aggregation_checksum=digest(POLICY),
            requirements=[
                dict(
                    id=i["id"],
                    definition=i["definition"],
                    input_modalities=["SLO"],
                    basis_kind="qualitative",
                )
                for i in policy["items"]
            ],
        )
        policy["items"] = [i for i in policy["items"] if i["observation_code"] in available]
        policy = validate(TaskPolicyV3, policy)
        raw["material"]["observations"] = [
            o for o in raw["material"]["observations"] if o["reference"]["code"] in available
        ]
        raw["policy"] = policy
        raw["material"]["scope"]["policy_checksum"] = digest(policy)
        raw["material"]["policy_ref"]["checksum"] = digest(policy)
        raw["material"]["scope"].pop("material_checksum")
        raw["material"]["scope"]["material_checksum"] = digest(raw["material"])
        return validate(DomainState, raw), image

    monkeypatch.setattr(module, "initial_state", initial)
    f = module.assembly(
        monkeypatch, count=11, replies=[batch_reply, Reply(content=b"", error="timeout")]
    )
    results, _ = run_case_stages(
        planner=f.planner,
        owner=f.owner,
        ports=f.ports,
        review=f.review,
        documents=f.documents,
        binding=f.binding,
        config=f.config,
        output=tmp_path,
        signal_cancel=Event(),
        selected_roles=("review",),
    )
    assert results[-1]["domain_status"] == "accepted", results
    scores = f.review.acceptance.read(key=results[-1]["acceptance_key"])[2]["scores"]
    assert len(scores["reviews"]) == 3
    assert len(scores["review_completion"]["pending_ids"]) == 2
    assert len(f.wire.calls) == 2


@pytest.mark.parametrize("revision", [None, "startup-r1"])
@pytest.mark.parametrize("configuration", [None, "selected-deployment.json"])
def test_model_read_uses_original_output_owner_not_visible_store(
    tmp_path, monkeypatch, revision, configuration
):
    from apps.vectorengine_single_case import write_once
    from packages.evaluation_preparation import dynamic_model as m
    from packages.fairclip_research.core import CONFIG_CHECKSUM, WEIGHT_SHA256, scores_from_cosines
    from packages.grounding.models import canonical
    from packages.research_runtime import deployment as d

    image = digest("image")
    normalized, index = scores_from_cosines((0.1, 0.2))
    output = dict(
        schema_version="fairclip-output/1.0.0",
        input_checksum=image,
        model_checksum="sha256:" + WEIGHT_SHA256,
        config_checksum=CONFIG_CHECKSUM,
        raw_cosines=[0.1, 0.2],
        normalized_model_scores=list(normalized),
        predicted_index=index,
        predicted_class="glaucoma",
        calibration="unverified",
        research_only=True,
        decisive_evidence=False,
        demographics_used=[],
    )
    raw = canonical(output)
    source = dict(
        artifact_code="ASSET:MODEL",
        artifact_checksum=checksum(raw),
        qualification="research_output",
        synthetic=False,
    )
    report = dict(model_outputs=[dict(output=output, source=source)])
    request = dict(lock=dict(run_id="RUN:MODEL"))
    context = SimpleNamespace(
        run_id="RUN:MODEL",
        request=SimpleNamespace(
            scope=SimpleNamespace(case_code="CASE:NEW"), model_dump=lambda **kw: request
        ),
    )
    repo = SimpleNamespace(
        read_research_run=lambda _: context,
        lifecycle=lambda _: dict(phase="FINISHED", report=report),
    )
    monkeypatch.setattr(m, "ResearchLifecycleRepository", lambda _: repo)
    configurations = []

    def load_configuration(path):
        configurations.append(str(path))
        return dict(installed_bundle="original", checksum=digest("deployment"))

    monkeypatch.setattr(d, "load_configuration", load_configuration)
    monkeypatch.setattr(d, "read_installed_bundle", lambda _: dict(output_root="original-output"))
    roots = []

    def store(root, kind):
        roots.append((root, kind))
        return SimpleNamespace(read_bounded=lambda uri, bound: raw)

    monkeypatch.setattr(d, "existing_store", store)
    directory = tmp_path / ("fairclip-" + revision if revision else "fairclip-v1")
    directory.mkdir()
    write_once(directory / "request.json", request)
    write_once(directory / "result.json", report)
    asset = SimpleNamespace(uri="object://knowledge/objects/test", checksum=checksum(raw))
    base = SimpleNamespace(
        directory=tmp_path,
        binding=dict(case_code="CASE:NEW", sample_id="random.npz"),
        runtime=SimpleNamespace(read_case_artifact=lambda *args: asset),
        image_artifact=SimpleNamespace(checksum=image),
        objects=object(),
    )
    if configuration:
        write_once(
            directory / "binding.json",
            dict(
                schema_version="new-case-fairclip-binding/1.0.0",
                case_code="CASE:NEW",
                sample_id="random.npz",
                request_checksum=digest(request),
                image_checksum=image,
                result_checksum=digest(report),
                configuration_checksum=digest("deployment"),
                original_executor=True,
            ),
        )
    assert (
        m.read_model(base, None, revision=revision, configuration=configuration)["output"] == output
    )
    assert configurations == [configuration or "configs/research/fairclip_single_case.json"]
    assert roots == [("original-output", "research_output")]
    base.image_artifact.checksum = digest("different image")
    with pytest.raises(ValueError, match="dynamic_model_.*binding"):
        m.read_model(base, None, revision=revision, configuration=configuration)
    if configuration:
        base.image_artifact.checksum = image
        monkeypatch.setattr(
            d,
            "load_configuration",
            lambda _: dict(installed_bundle="original", checksum=digest("changed deployment")),
        )
        with pytest.raises(ValueError, match="dynamic_model_selected_configuration_binding"):
            m.read_model(base, None, revision=revision, configuration=configuration)


def test_canonical_model_observation_fits_bounded_projection():
    from packages.evaluation_preparation.dynamic_model import attach_model

    f = dynamic_fixture(1, empty=True)
    model = dict(
        output=dict(
            normalized_model_scores=[0.4, 0.6],
            raw_cosines=[0.1, 0.2],
            input_checksum=digest("image"),
        ),
        source=dict(
            artifact_code="ASSET:MODEL",
            artifact_checksum=digest("output"),
            implementation={},
            long_provenance="x" * 3000,
        ),
        run_id="RUN:MODEL",
    )
    result = attach_model(f.owner.state, model)
    assert len(result.material.observations[0].value) <= 1000
