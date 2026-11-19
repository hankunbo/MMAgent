"""Structured matching; text ranking never grants applicability or current authority."""

from dataclasses import dataclass

from packages.evidence_details.current_rejection import current_rejection
from packages.grounding.models import digest
from packages.memory_review.models import Query
from packages.memory_review.task_models import TaskBinding, TaskQuery, qualification


@dataclass(frozen=True)
class Retrieval:
    query: Query
    groups: dict
    rejected: tuple
    status: str = "ok"
    error: str | None = None

    def document(self):
        return dict(
            schema_version=(
                "memory-retrieval/1.1.0"
                if isinstance(self.query, TaskQuery)
                else "memory-retrieval/1.0.0"
            ),
            query=self.query.model_dump(mode="json"),
            groups=self.groups,
            rejected=self.rejected,
            status=self.status,
            error=self.error,
        )


def mismatch(binding, query):
    if binding.group == "success":
        if isinstance(binding, TaskBinding) != isinstance(query, TaskQuery):
            return "memory_success_semantics"
        if isinstance(binding, TaskBinding) and binding.task_profile != query.task_profile:
            return "memory_task_profile"
    c = binding.conditions
    for field in ("goal", "template", "domain", "validation"):
        if getattr(c, field) != getattr(query, field):
            return "memory_match_" + field
    for field in ("graph_conditions", "modalities", "inputs", "failure_types"):
        required, actual = getattr(c, field), getattr(query, field)
        if "unknown" in required or not set(required) <= set(actual):
            return "memory_match_" + field
    for field in ("patient_conditions", "tool_versions", "rule_versions"):
        actual = getattr(query, field)
        if any(
            v == "unknown" or actual.get(k) in {None, "unknown"} or actual[k] != v
            for k, v in getattr(c, field).items()
        ):
            return "memory_match_" + field
    return None


# Only explicit business refusals are filters. Unknown ValueError/I/O remain technical.
VERSION_REFUSALS = {
    "memory_tool_expired",
    "memory_tool_not_currently_admitted",
    "memory_tool_version_changed",
    "memory_rule_version_changed",
}

REFUSALS = {
    "memory_qualification_revoked",
    "memory_qualification_expired",
    "memory_parent_invalid",
    "memory_source_withdrawn",
    "memory_read_changed",
    "memory_distillation_binding",
    "memory_publication_policy_unapproved",
    "memory_synthetic_required",
    "memory_opinion_expiry",
    "memory_independent_review_required",
    "memory_source_binding",
    "memory_source_version",
}


class MemoryRetrieval:
    def __init__(self, *, repository, query_reader, optional=True):
        self.repository, self.query_reader, self.optional = repository, query_reader, optional

    def query(self, run_id):
        q = qualification(Query, self.query_reader.current(run_id))
        context = self.repository.contexts.read_research_run(run_id)
        if not context.synthetic or (q.run_id, q.case_code) != (
            run_id,
            context.request.scope.case_code,
        ):
            raise ValueError("memory_query_scope")
        return q

    @staticmethod
    def _hit(p, r, score):
        hit = dict(
            library="memory",
            group=p.binding.group,
            publication=p.key,
            publication_checksum=digest(p),
            target=r.key,
            content_version=r.schema_version,
            checksum=digest(r),
            sources=tuple(sorted(digest(s) for s in r.sources)),
            cases=tuple(sorted({s.case_code for s in r.sources})),
            policy=p.policy_key,
            policy_checksum=p.policy_checksum,
            branches=p.binding.preferred_branches,
            text_score=score,
            reason="all_required_conditions_matched",
            clinical_level="not_evaluated",
            qualification="synthetic_offline_only",
        )

        if isinstance(p.binding, TaskBinding):
            hit.update(
                success_semantics=p.binding.success_semantics,
                task_profile=p.binding.task_profile,
                task_proofs=tuple(proof.checksum for proof in p.binding.proofs),
            )
        return hit

    def _independent_query(self, binding, query):
        if not isinstance(binding, TaskBinding):
            return None
        current = self.repository.contexts.read_research_run(query.run_id).request.scope
        for proof in binding.proofs:
            support = self.repository.contexts.read_research_run(
                proof.definition.run_id
            ).request.scope
            if (
                current.case_code == support.case_code
                or current.patient_hash == support.patient_hash
            ):
                return "memory_task_query_support_overlap"
        return None

    def recheck(self, query, hits):
        if self.query(query.run_id) != query:
            raise ValueError("memory_query_changed")
        for hit in hits:
            p = self.repository.read_publication(hit["publication"])
            if (
                digest(p) != hit["publication_checksum"]
                or mismatch(p.binding, query)
                or self._independent_query(p.binding, query)
            ):
                raise ValueError("memory_retrieval_changed")
            if (p.test_context, p.purpose) != (query.test_context, query.purpose):
                raise ValueError("memory_publication_purpose")
            if p.binding.group not in query.groups:
                raise ValueError("memory_group_disabled")
            record = self.repository.read(p.binding.target).record
            expected = self._hit(p, record, len(set(query.text) & set(p.binding.terms)))
            if digest(hit) != digest(expected):
                raise ValueError("memory_hit_binding")
            self.query_reader.check_versions(query)

    def retrieve(self, run_id):
        q = self.query(run_id)
        groups = {k: [] for k in ("success", "counterfactual", "recovery")}
        rejected, ranked = [], []
        try:
            try:
                self.query_reader.check_versions(q)
            except ValueError as error:
                if type(error) is not ValueError or str(error) not in VERSION_REFUSALS:
                    raise
                return Retrieval(q, groups, (dict(reason=str(error)),), "not_applicable")
            for key in self.repository.publication_keys():
                try:
                    p = self.repository.read_publication(key)
                except Exception as e:
                    if current_rejection(e) or (type(e) is ValueError and str(e) in REFUSALS):
                        rejected.append(dict(publication=key, reason=str(e)))
                        continue
                    raise
                reason = mismatch(p.binding, q) or self._independent_query(p.binding, q)
                if (p.test_context, p.purpose) != (q.test_context, q.purpose):
                    reason = "memory_publication_purpose"
                if p.binding.group not in q.groups:
                    reason = "memory_group_disabled"
                if reason:
                    rejected.append(dict(publication=key, reason=reason))
                    continue
                record = self.repository.read(p.binding.target).record
                score = len(set(q.text) & set(p.binding.terms))
                ranked.append((p, record, score))
            used_cases, used_sources = set(), set()
            for p, r, score in sorted(ranked, key=lambda v: (-v[2], v[0].key)):
                # Overlap may be informative in a different group, never repeated votes.
                cases = {(p.binding.group, s.case_code) for s in r.sources}
                sources = {(p.binding.group, digest(s)) for s in r.sources}
                if cases & used_cases or sources & used_sources:
                    rejected.append(dict(publication=p.key, reason="memory_duplicate_support"))
                    continue
                used_cases.update(cases)
                used_sources.update(sources)
                groups[p.binding.group].append(self._hit(p, r, score))
            self.recheck(q, [h for g in groups.values() for h in g])
            return Retrieval(q, groups, tuple(rejected))
        except Exception as error:
            if not self.optional:
                raise
            # Explicit technical status, never an apparently successful empty library.
            code = (
                str(error)
                if type(error) is ValueError and str(error) in REFUSALS
                else "memory_retrieval_unavailable"
            )
            return Retrieval(q, {k: [] for k in groups}, tuple(rejected), "technical_failure", code)


class JointRetrieval:
    """Keep independently authorized Knowledge and Memory results in separate fields."""

    def __init__(self, *, knowledge, memory):
        self.knowledge, self.memory = knowledge, memory

    def retrieve(self, *, run_id, knowledge_query, knowledge_context, knowledge_config):
        knowledge = self.knowledge.retrieve_published_knowledge(
            run_id, knowledge_query, knowledge_context, knowledge_config
        )
        return dict(
            knowledge=knowledge,
            memory=self.memory.retrieve(run_id),
            roles=dict(
                knowledge="medical_rules", memory="planning_prior", tool="06_current_admission"
            ),
        )
