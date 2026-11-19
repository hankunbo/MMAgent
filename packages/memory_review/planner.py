"""Explicit 10D installation on original RankedActions/Planner; additive Runtime receipts."""

from packages.grounding.models import digest, validate
from packages.memory_review.models import Query
from packages.memory_review.task_models import qualification
from packages.planner_control.actions import RankedActions, baseline
from packages.planner_control.models import ActionDecision, SelectionOutcome, TraceRanking


class MemoryRankedActions(RankedActions):
    def __init__(self, source, *, retrieval, documents, policy=None, agent=None):
        super().__init__(source, policy=policy, agent=agent)
        self.retrieval, self.documents = retrieval, documents

    def _key(self, state, decision=None):
        if state.binding.run_id != self.documents.run_id:
            raise ValueError("memory_decision_run_binding")
        if state.pending and state.pending.command.action == "select":
            step = state.pending.command.key
        else:
            matches = [
                e
                for e in state.events
                if isinstance(e.outcome, SelectionOutcome) and e.outcome.decision == decision
            ]
            if not matches:
                raise ValueError("memory_decision_step_binding")
            step = matches[-1].command.key
        return "memory-selection:" + step

    def read_influence(self, state, decision=None):
        values = self.documents.read(self._key(state, decision))
        if (
            values is None
            or len(values) != 1
            or values[0].get("schema_version")
            not in {"memory-selection/1.0.0", "memory-selection/1.1.0"}
        ):
            raise ValueError("memory_decision_missing")
        result = values[0]
        query = result["retrieval"].get("query")
        task = query is not None and query.get("schema_version") == "memory-query/1.1.0"
        expected_version = "memory-selection/1.1.0" if task else "memory-selection/1.0.0"
        if result["schema_version"] != expected_version:
            raise ValueError("memory_selection_semantics")
        saved = validate(ActionDecision, result["decision"])
        if decision is not None and saved != decision:
            raise ValueError("memory_decision_binding")
        if (
            result["run_id"] != state.binding.run_id
            or result["action_set_checksum"] != saved.action_set.checksum
        ):
            raise ValueError("memory_decision_binding")
        return result

    def select(self, state, clock):
        key = self._key(state)
        prior = self.documents.read(key)
        if prior:
            decision = validate(ActionDecision, self.read_influence(state)["decision"])
            return self.resume_selection(state, decision, clock)
        base = super().select(state, clock)
        a = base.action_set
        deterministic = baseline(a, legacy=self.policy.fallback == "legacy_id_baseline")
        try:
            result = self.retrieval.retrieve(state.binding.run_id)
            retrieval = result.document()
        except Exception:
            if not self.retrieval.optional:
                raise
            retrieval = dict(
                schema_version="memory-retrieval/1.0.0",
                query=None,
                groups={k: [] for k in ("success", "counterfactual", "recovery")},
                rejected=[],
                status="technical_failure",
                error="memory_retrieval_unavailable",
            )
        hits = [h for group in retrieval["groups"].values() for h in group]
        influence = {action.id: 0 for action in a.actions}
        # Deduplicate again across groups: the same cases cannot amplify their vote.
        used = set()
        for hit in hits:
            if set(hit["cases"]) & used:
                continue
            used.update(hit["cases"])
            for action in a.actions:
                if action.candidate is not None and action.branch in hit["branches"]:
                    influence[action.id] = max(influence[action.id], 1)
        order = tuple(
            sorted(
                base.ranking.ranked_ids,
                key=lambda i: (-influence[i], base.ranking.ranked_ids.index(i)),
            )
        )
        decision = base
        if any(influence.values()):
            decision = validate(
                ActionDecision,
                base.model_copy(
                    update=dict(
                        ranking=TraceRanking(
                            set_checksum=a.checksum,
                            policy_checksum=digest(self.policy),
                            ranked_ids=order,
                            reason="memory_synthetic_prior_receipt:" + digest(key)[7:],
                        ),
                        strategy="trace-ranking/1.0.0",
                        selected_id=order[0],
                    )
                ),
            )
        if self.current(state) != a or clock() >= a.expires_at:
            raise ValueError("stale_action_set")
        if hits:
            self.retrieval.recheck(qualification(Query, retrieval["query"]), hits)
        receipt = dict(
            schema_version=(
                "memory-selection/1.1.0"
                if retrieval.get("query") is not None
                and retrieval["query"]["schema_version"] == "memory-query/1.1.0"
                else "memory-selection/1.0.0"
            ),
            run_id=state.binding.run_id,
            step_key=state.pending.command.key,
            request_checksum=state.pending.request_checksum,
            action_set_checksum=a.checksum,
            graph_checksum=a.graph.checksum,
            budget=a.budget.model_dump(mode="json"),
            retrieval=retrieval,
            baseline_order=deterministic.ranked_ids,
            pre_memory_order=base.ranking.ranked_ids,
            influence=influence,
            final_order=decision.ranking.ranked_ids,
            selected_id=decision.selected_id,
            decision=decision.model_dump(mode="json"),
        )
        with self.documents.session():
            self.documents.write(key, (receipt,))
        return decision

    def confirm_selection(self, state, decision, clock):
        receipt = self.read_influence(state, decision)
        if self.current(state) != decision.action_set or clock() >= decision.action_set.expires_at:
            raise ValueError("stale_pending_selection")
        if (
            state.pending
            and state.pending.command.action == "select"
            and (
                receipt["step_key"] != state.pending.command.key
                or receipt["request_checksum"] != state.pending.request_checksum
            )
        ):
            raise ValueError("memory_decision_step_binding")
        self._current_memory(receipt)

    def _current_memory(self, receipt):
        result = receipt["retrieval"]
        hits = [h for group in result["groups"].values() for h in group]
        if hits:
            self.retrieval.recheck(qualification(Query, result["query"]), hits)

    def resume_selection(self, state, decision, clock):
        decision = super().resume_selection(state, decision, clock)
        self.confirm_selection(state, decision, clock)
        return decision

    def recheck(self, state, decision, clock):
        super().recheck(state, decision, clock)
        self._current_memory(self.read_influence(state, decision))
