"""Real saved text (offline) and injected HTTP; never sends remote requests."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from packages.knowledge_claims.research_candidates import validate_research_source

LAYOUT_POLICY = "research-quote-policy/layout-whitespace/1.0.0"


def real_failure():
    package = Path("docs/acceptance/12-completion-cde-target-live-20261001")
    request = json.loads((package / "actual-requests.json").read_text())[0]
    candidate = json.loads(request["raw_response"][0]["raw_model_text"])
    documents = json.loads((package / "original-case-documents.json").read_text())
    intent = next(
        v for k, v in documents.items() if k.startswith("review:") and k.endswith(":intent")
    )
    return candidate, intent[0]["state"]["material"]


def test_original_failure_accepted_only_under_explicit_layout_policy():
    candidate, material = real_failure()
    before = json.dumps([candidate, material], sort_keys=True)
    with pytest.raises(ValueError, match="research_source_or_target_mismatch"):
        validate_research_source(candidate, material)
    result = validate_research_source(candidate, material, quote_policy=LAYOUT_POLICY)
    assert result.model_dump(mode="json") == candidate
    assert json.dumps([candidate, material], sort_keys=True) == before


def prepared(*, policy=LAYOUT_POLICY, quote=None):
    from agents.provider_agents import StructuredRoleAgent
    from packages.evidence_details.acceptance import ResearchAcceptance
    from packages.grounding.models import digest, validate
    from packages.llm_provider import DeterministicFakeProvider
    from packages.llm_provider.config import load_provider_config
    from packages.llm_provider.models import SourceProposal
    from tests.agent_provider.support import Documents, budget_for, materials, submission

    config = load_provider_config(environ={})
    reader, binding = materials(role="source", config=config)
    reader.source_quote_policy = policy
    text = reader.state.material.sources[0].text
    candidate = validate(
        SourceProposal,
        dict(
            scope_checksum=digest(reader.state.material.scope),
            source_code="SOURCE:ONE",
            quote=quote if quote is not None else text.replace(" ", "\n", 1),
            target=reader.state.material.goal_code,
            relation="requires",
            missing=[],
            reason="Synthetic citation fidelity only",
        ),
    )
    provider = DeterministicFakeProvider(config=config, replies=[candidate.model_dump(mode="json")])
    budget = budget_for(binding, config)
    agent = StructuredRoleAgent("source", provider=provider, materials=reader, budget=budget)
    documents = Documents()
    service = ResearchAcceptance(owner=reader, documents=documents, requests=budget)
    return SimpleNamespace(
        reader=reader,
        binding=binding,
        candidate=candidate,
        agent=agent,
        provider=provider,
        budget=budget,
        documents=documents,
        service=service,
        submission=lambda c: submission(reader, c, budget),
    )


def generated(f):
    c = f.agent.call(f.reader.state.material.scope, deadline=f.binding.deadline)
    return f.submission(c)


def test_provider_domain_and_report_share_policy_preserve_original_and_replay():
    from agents.expert_agent.reporting import read_research_report, render_research_html

    f = prepared()
    sub = generated(f)
    first = f.service.submit(sub, key="aligned")
    assert first["schema_version"] == "research-submission-receipt/1.1.0"
    proof = first["source_quote_validation"]
    assert proof["alignment"]["mode"] == "layout_whitespace"
    assert first["submission"]["candidate"]["quote"] == f.candidate.quote
    receipt = f.service.accept(key="aligned")
    assert receipt.status == "accepted" and receipt.scores_checksum is None
    report = read_research_report(f.service, key="aligned", parent=f.reader.state)
    assert report["schema_version"] == "research-successor-report/1.1.0"
    assert report["source_quote_validation"] == proof
    assert "layout_whitespace" in render_research_html(report)
    before = list(f.documents.audit)
    assert f.service.submit(sub, key="aligned") == first
    assert f.service.accept(key="aligned") == receipt
    f.reader.withdrawn = True
    assert read_research_report(f.service, key="aligned", parent=f.reader.state) == report
    assert f.service.accept(key="aligned") == receipt  # historical, not new authorization
    assert f.documents.audit == before and f.budget.used == 1


@pytest.mark.parametrize("change", ["proof", "policy", "withdrawn", "source_text"])
def test_domain_acceptance_rechecks_original_material_and_policy(change):
    from packages.knowledge_claims.research_quote import LITERAL_POLICY

    f = prepared()
    sub = generated(f)
    f.service.submit(sub, key="one")
    if change == "proof":
        f.documents.saved["submission:one"][0]["source_quote_validation"]["alignment"]["start"] += 1
    elif change == "policy":
        f.reader.source_quote_policy = LITERAL_POLICY
    elif change == "withdrawn":
        f.reader.withdrawn = True
    else:
        original = f.reader.state.material.sources[0]
        f.reader.state = f.reader.state.model_copy(
            update={
                "material": f.reader.state.material.model_copy(
                    update={"sources": (original.model_copy(update={"text": "Withdrawn text"}),)}
                )
            }
        )
    if change in {"withdrawn", "source_text"}:
        with pytest.raises(ValueError):
            f.service.accept(key="one")
    else:
        assert f.service.accept(key="one").status == "rejected"
    saved = f.documents.read("accepted:one")
    assert saved is None or saved[-1]["status"] == "rejected"
    assert f.budget.used == 1


def test_direct_domain_cannot_bypass_quote_policy_with_successful_provider_record():
    from packages.llm_provider.models import SourceProposal

    f = prepared(quote="Synthetic source refutes a bounded research comparison.")
    # Bypass role-admission deliberately; original domain must still reject.
    candidate = f.provider.generate_structured(
        role="source",
        input_payload=dict(role="source", material=f.reader.state.material.model_dump(mode="json")),
        output_schema=SourceProposal,
        scope=f.reader.state.material.scope,
        budget=f.budget,
        deadline=f.binding.deadline,
    )
    assert f.service.submit(f.submission(candidate), key="bad").status == "rejected"


@pytest.mark.parametrize(
    "policy", [None, "layout", "research-quote-policy/layout-whitespace/9.0.0"]
)
def test_unrecognized_policy_rejected(policy):
    from packages.knowledge_claims.research_quote import check_quote_policy

    with pytest.raises(ValueError, match="policy_not_supported"):
        check_quote_policy(policy)


def test_original_literal_owner_and_report_remain_exact_version():
    from agents.expert_agent.reporting import read_research_report
    from packages.knowledge_claims.research_quote import LITERAL_POLICY
    from packages.llm_provider.provider import ProviderError

    f = prepared(policy=LITERAL_POLICY)
    with pytest.raises(ProviderError) as exc:
        generated(f)
    assert exc.value.code == "scope" and f.budget.used == 1
    g = prepared(policy=LITERAL_POLICY, quote=f.reader.state.material.sources[0].text)
    sub = generated(g)
    assert g.service.submit(sub, key="old")["schema_version"] == "research-submission-receipt/1.0.0"
    assert g.service.accept(key="old").status == "accepted"
    report = read_research_report(g.service, key="old", parent=g.reader.state)
    assert report["schema_version"] == "research-successor-report/1.0.0"
    assert "source_quote_validation" not in report


def test_html_proof_tampering_and_markup_escape():
    import copy

    from agents.expert_agent.reporting import read_research_report, render_research_html

    f = prepared()
    f.service.submit(generated(f), key="one")
    f.service.accept(key="one")
    report = read_research_report(f.service, key="one", parent=f.reader.state)
    for field, value in [("policy", "other"), ("request_id", "other")]:
        altered = copy.deepcopy(report)
        altered["source_quote_validation"][field] = value
        with pytest.raises(ValueError, match="quote_binding"):
            render_research_html(altered)
    report["extra_display"] = "<script>bad</script>"
    assert "<script>bad</script>" not in render_research_html(report)


def wrapped_source(body):
    from tests.expert_agent.batch_case_support import source_reply

    reply = source_reply(body)
    reply["quote"] = reply["quote"].replace(" ", "\n", 1)
    return reply


def test_injected_http_three_stages_multi_receipt_score_and_html(monkeypatch, tmp_path):
    from tests.expert_agent.batch_case_support import REPLIES
    from tests.expert_agent.test_batch_assembly import assembly, stages

    f = assembly(
        monkeypatch,
        replies=[wrapped_source, *REPLIES[1:]],
        source_quote_policy=LAYOUT_POLICY,
        config_path="configs/agents/vectorengine_batch_citations.json",
    )
    completed, roles = stages(f, tmp_path)
    assert roles == ("source", "vision", "review")
    assert all(r["status"] == "accepted" for r in completed)
    report = json.loads((tmp_path / "source-report.json").read_text())
    assert report["source_quote_validation"]["alignment"]["mode"] == "layout_whitespace"
    assert (
        report["records"]["candidate"]["quote"]
        != report["source_quote_validation"]["alignment"]["original_text"]
    )
    assert "计算依赖与判读背景" in (tmp_path / "review-report.html").read_text()
    scores = json.loads((tmp_path / "review-report.json").read_text())["scores"]
    assert scores["schema_version"] == "case-research-scoring/6.1.0"
    assert f.ports.budget.used == len(f.wire.calls) == 5
    audit = list(f.documents.audit)
    assert stages(f, tmp_path)[0] == completed and f.documents.audit == audit
