"""Synthetic tasks and wire replies; no patient objects, credentials or sockets."""

import json
from types import SimpleNamespace

from agents.provider_agents import StructuredRoleAgent
from agents.provider_agents.batched_review import run_batches
from apps.vectorengine_single_case import load_config
from packages.evidence_details.acceptance import ResearchAcceptance
from packages.llm_provider import DeterministicFakeProvider
from packages.llm_provider.public_shape import PublicReviewMaterials
from tests.agent_provider.support import Documents, budget_for, review_from_wire
from tests.planner.support import binding

CONFIG = "configs/agents/vectorengine_batch_review.json"


def batch_reply(body):
    raw = review_from_wire(body)
    payload = json.loads(body["messages"][1]["content"][0]["text"])
    raw.update(
        schema_version="research-review-batch/1.0.0",
        plan_checksum=payload["batch"]["plan_checksum"],
        batch_id=payload["batch"]["batch_id"],
    )
    tasks = {t["assessment_id"]: t for t in payload["review_tasks"]["tasks"]}
    for row in raw["items"]:
        if (
            row["status"] == "assessed"
            and tasks[row["assessment_id"]]["definition"]["interpretation_status"] == "missing"
        ):
            row.update(
                status="reference_missing",
                judgment=dict(
                    value=None,
                    lower=None,
                    upper=None,
                    reason="Synthetic reference absent",
                    method="not_assessed",
                    missing_reason="Synthetic definition missing",
                ),
            )
    return raw


def fixture(count=7, *, replies=None, documents=None, config_path=CONFIG):
    config = load_config(config_path).model_copy(update={"mode": "fake"})
    b = binding()
    owner = PublicReviewMaterials(b.run_id, config, task_count=count)
    budget = budget_for(b, config)
    provider = DeterministicFakeProvider(config, replies or [batch_reply] * 15)
    docs = documents or Documents()
    agent = StructuredRoleAgent("review", provider=provider, materials=owner, budget=budget)
    acceptance = ResearchAcceptance(owner=owner, documents=docs, requests=budget)
    return SimpleNamespace(
        config=config,
        binding=b,
        owner=owner,
        budget=budget,
        provider=provider,
        documents=docs,
        agent=agent,
        acceptance=acceptance,
    )


def generate(f, key="synthetic-batches"):
    return run_batches(
        f.agent, state=f.owner.current(), documents=f.documents, key=key, round_id="A"
    )
