"""Fake Memory advice: shared 08 budget, per-attempt admission, immutable recovery."""

from packages.grounding.models import digest, validate
from packages.memory_review.models import Advice, Binding, Opinion, ReviewInput
from packages.memory_review.task_models import (
    TaskBinding,
    TaskOpinion,
    TaskReviewInput,
    qualification,
)


class MemoryReviewAgent:
    def __init__(self, *, repository, provider, budget, documents, scope):
        self.repository, self.provider, self.budget = repository, provider, budget
        self.documents, self.scope = documents, scope

    def review(self, *, key, binding, valid_from, expires_at):
        binding = qualification(Binding, binding)
        r = self.repository._target(binding)
        input_model = TaskReviewInput.project if isinstance(binding, TaskBinding) else ReviewInput
        opinion_model = TaskOpinion if isinstance(binding, TaskBinding) else Opinion
        payload = input_model(
            scope=self.scope.model_dump(mode="json"),
            binding=binding,
            sources=tuple(sorted(digest(s) for s in r.record.sources)),
        )
        request = dict(
            input=payload.model_dump(mode="json"),
            valid_from=valid_from.isoformat(),
            expires_at=expires_at.isoformat(),
        )
        request_key, result_key = "memory-review-request:" + key, "memory-review-result:" + key
        with self.documents.session():
            old = self.documents.read(request_key)
            saved = self.documents.read(result_key)
            if old is not None and old != (request,):
                raise ValueError("memory_key_content_conflict")
            if saved:
                opinion = qualification(Opinion, saved[0])
                return self.repository.record_opinion(opinion)
            if old:
                raise ValueError("memory_review_pending_unknown")
            self.documents.write(request_key, (request,))

        def before_request():
            self.documents.validate_scope(self.scope)
            current = self.repository._target(binding)
            self.repository._synthetic(current.record)
            if self.budget.clock() >= min(expires_at, self.budget.deadline):
                raise ValueError("memory_review_deadline")
            if tuple(sorted(digest(s) for s in current.record.sources)) != payload.sources:
                raise ValueError("memory_opinion_sources")

        def admit(result):
            before_request()
            if (result.target_checksum, result.binding_checksum) != (
                binding.target_checksum,
                digest(binding),
            ):
                raise ValueError("memory_advice_binding")

        before = self.budget.used
        result = self.provider.generate_memory_review(
            input_payload=payload,
            scope=self.scope,
            budget=self.budget,
            deadline=min(expires_at, self.budget.deadline),
            before_request=before_request,
            admit=admit,
        )
        result = validate(Advice, result)
        admit(result)
        requests = tuple(digest(r) for r in self.budget.records() if r.attempt > before)
        opinion = opinion_model(
            key=key,
            binding=binding,
            kind="agent",
            decision=result.decision,
            reasons=result.reasons,
            source_checksums=payload.sources,
            valid_from=valid_from,
            expires_at=expires_at,
            request_checksums=requests,
        )
        with self.documents.session():
            self.documents.write(result_key, (opinion,))
        return self.repository.record_opinion(opinion)
