"""Explicit synthetic research-run binding to an authentic Published Knowledge release.

The old formal Runtime Knowledge lock is not widened or forged. This independent
Runtime document is only usable for synthetic runs and synthetic published releases.
"""

from apps.knowledge_retrieval import PublishedKnowledgeService
from db.knowledge.models import ApplicabilityContext
from packages.grounding.models import digest, validate
from packages.memory_review.models import JointKnowledgeBinding


class SyntheticPublishedKnowledge:
    version = "memory-joint-knowledge-binding/1.0.0"
    key = "memory-joint-knowledge-binding"

    def __init__(self, *, documents, repository, object_store):
        self.documents, self.repository = documents, repository
        self.published = PublishedKnowledgeService(repository, object_store)

    def _synthetic(self, run_id, release, query_context):
        if run_id != self.documents.run_id:
            raise ValueError("memory_joint_run_binding")
        context = self.documents.contexts.read_research_run(run_id)
        eligibility = ApplicabilityContext(
            **{field: getattr(query_context, field) for field in ApplicabilityContext.model_fields}
        )
        current = self.repository.get_published(release, applicability=eligibility)
        if not context.synthetic or not current.record.synthetic:
            raise ValueError("memory_joint_synthetic_required")
        return context

    def bind(self, release, *, query, context, config):
        run_id = self.documents.run_id
        original = self._synthetic(run_id, release, context)
        self.published.retrieve_published_knowledge(release, query, context, config)
        value = JointKnowledgeBinding(
            schema_version=self.version,
            run_id=run_id,
            run_context_checksum=original.checksum,
            release=release,
            context_checksum=digest(dict(vars(context))),
            config_checksum=digest(dict(vars(config))),
        )
        with self.documents.session():
            self.documents.write(self.key, (value,))
        return value

    def retrieve_published_knowledge(self, run_id, query, context, config):
        values = self.documents.read(self.key)
        if not values or len(values) != 1 or values[0].get("schema_version") != self.version:
            raise ValueError("memory_joint_knowledge_not_bound")
        bound = validate(JointKnowledgeBinding, values[0])
        value = bound.model_dump(mode="json")
        release = bound.release
        original = self._synthetic(run_id, release, context)
        if (
            value["run_id"],
            value["run_context_checksum"],
            value["context_checksum"],
            value["config_checksum"],
        ) != (run_id, original.checksum, digest(dict(vars(context))), digest(dict(vars(config)))):
            raise ValueError("memory_joint_knowledge_binding")
        result = self.published.retrieve_published_knowledge(release, query, context, config)
        return dict(library="knowledge", role="medical_rules", binding=value, result=result)
