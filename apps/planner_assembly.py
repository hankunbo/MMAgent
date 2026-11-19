"""Explicit research consumer assembly. No live execution is installed by default."""

import json

from agents.planner.service import Planner
from apps.evidence_graph_records import inspect
from apps.evidence_graph_scores import unscored
from apps.glaucoma_case_storage import read_view
from packages.evidence_scoring.api import build, validate
from packages.grounding.models import digest
from packages.object_store import checksum
from packages.planner_control.models import Graph, Outcome, Reference
from packages.research_runtime.planner_steps import ResearchPlannerSteps


class ResearchSnapshotSteps:
    """One graph owner; detect any changed graph inventory before using a local snapshot.

    inventory_checksum binds all registered graph artifacts at assembly time, including
    historical versions. New external writers force reassembly/revalidation, never 'latest'.
    """

    def __init__(self, *, visible, repository, objects, inventory_checksum, locks):
        self.visible, self.repository, self.objects = visible, repository, objects
        self.inventory_checksum = inventory_checksum
        self.locks = tuple(locks)

    def _read(self, binding):
        if binding.graph.kind != "research_artifact":
            raise ValueError("formal_graph_requires_original_03_admission")
        if binding.locks != self.locks:
            raise ValueError("installed_version_locks_mismatch")
        visible = self.visible.read(binding.case_code)
        if (
            visible.checksum != binding.input_checksum
            or digest(visible) != binding.visible_checksum
        ):
            raise ValueError("visible_input_changed")
        artifacts = self.repository.get_case_bundle(binding.case_code).artifacts
        graphs = sorted(
            (a.asset_code, a.checksum)
            for a in artifacts
            if a.artifact_type_code == "ARTIFACT:CASE_EVIDENCE_GRAPH"
        )
        if digest(graphs) != self.inventory_checksum:
            raise ValueError("graph_inventory_changed_reread_reassembly_required")
        artifact = next((a for a in artifacts if a.asset_code == binding.graph.code), None)
        if artifact is None or artifact.checksum != binding.graph.checksum:
            raise ValueError("exact_research_graph_required")
        raw = self.objects.read_bounded(artifact.uri, 4194304)
        if checksum(raw) != artifact.checksum:
            raise ValueError("graph_bytes_changed")
        manifest = json.loads(raw)
        if (
            manifest.get("schema") != binding.graph.version
            or manifest.get("schema") != "case-evidence-graph/1.19.0"
        ):
            raise ValueError("unsupported_research_graph_version")
        parent = binding.graph.parent
        if (
            parent is None
            or parent.version != "case-evidence-graph/1.18.0"
            or manifest.get("base") != dict(code=parent.code, checksum=parent.checksum)
        ):
            raise ValueError("exact_parent_graph_required")
        view = read_view(
            binding.case_code, artifact.asset_code, repo=self.repository, store=self.objects
        )
        return view

    def current(self, binding):
        self._read(binding)
        return Graph.model_validate_json(binding.graph.model_dump_json())

    def step(self, binding, action):
        view = self._read(binding)
        if action == "load":
            return Outcome(
                refs=(
                    Reference(
                        role="visible",
                        code=binding.case_code,
                        version="visible-input/1.0.0",
                        checksum=binding.input_checksum,
                    ),
                )
            )
        if action == "retrieve":
            return Outcome(
                reason="saved_sources_only_no_search",
                refs=tuple(
                    r
                    for r in binding.locks
                    if r.role in {"knowledge", "template", "rule", "sources"}
                ),
            )
        if action == "verify":
            base = unscored(view)
            score = build(base, version="5.0.0")
            validate(base, score)
            if score != view["current_scoring"]:
                raise ValueError("saved_v5_score_mismatch")
            # Source-owned record pointer validation; no new interpretation is generated.
            inspect(base, resolve_bindings=True)
            return Outcome(
                refs=(
                    Reference(
                        role="saved_v5",
                        code=binding.graph.code,
                        version="case-research-scoring/5.0.0",
                        checksum=score["checksum"],
                    ),
                )
            )
        return Outcome(
            refs=(
                Reference(
                    role="saved_research_graph",
                    code=binding.graph.code,
                    version=binding.graph.version,
                    checksum=binding.graph.checksum,
                ),
            )
        )


def assemble_research_planner(*, runtime, snapshots):
    return Planner(runtime=runtime, owner=ResearchPlannerSteps(snapshots=snapshots))


def assemble_ranked_research_planner(*, runtime, snapshots, policy=None, trace_agent=None):
    """08B opt-in; Binding must carry the exact RankingPolicy version/checksum."""
    from packages.planner_control.actions import RankedActions, SavedResearchActions

    return Planner(
        runtime=runtime,
        owner=ResearchPlannerSteps(snapshots=snapshots),
        ranking=RankedActions(SavedResearchActions(snapshots), policy=policy, agent=trace_agent),
    )
