"""Append-only synthetic REVIEW-001 exercise; never a release/publication authorization."""

from apps.api.auth import require_role
from apps.api.errors import Fault
from apps.api.models import OfflineVote
from packages.grounding.models import digest, validate


class OfflineReviewBoard:
    def __init__(self, *, run_id, case_code, documents, subject, read_subject, reviewers, arbiter):
        if (
            len(reviewers) not in (1, 2)
            or len(set(reviewers)) != len(reviewers)
            or arbiter in reviewers
        ):
            raise ValueError("offline_review_policy")
        self.run_id, self.case_code, self.documents = run_id, case_code, documents
        self.subject, self.read_subject = subject, read_subject
        self.reviewers, self.arbiter = tuple(reviewers), arbiter
        self.policy = dict(
            schema_version="synthetic-offline-review-policy/1.0.0",
            subject=subject.model_dump(mode="json"),
            reviewers=list(reviewers),
            arbiter=arbiter,
            real_review_001_approved=False,
            release_authorization=False,
        )

    def _current(self):
        if self.read_subject() != self.subject:
            raise Fault("VERSION_CONFLICT")

    def _key(self, actor):
        return "offline-vote:" + digest(dict(policy=self.policy, actor=actor))[7:]

    def read(self):
        # Historical votes remain readable after withdrawal. No current-use permission implied.
        with self.documents.session():
            votes = {
                actor: value[0]
                for actor in (*self.reviewers, self.arbiter)
                if (value := self.documents.read(self._key(actor))) is not None
            }
        decisions = [votes[a]["decision"] for a in self.reviewers if a in votes]
        if len(decisions) < len(self.reviewers):
            status, decision = "pending", None
        elif len(set(decisions)) == 1:
            status, decision = "reviewed", decisions[0]
        elif self.arbiter not in votes:
            status, decision = "adjudication_required", None
        else:
            status, decision = "adjudicated", votes[self.arbiter]["decision"]
        return dict(
            schema_version="synthetic-offline-review/1.0.0",
            run_id=self.run_id,
            case_code=self.case_code,
            policy=self.policy,
            votes=list(votes.values()),
            status=status,
            decision=decision,
            current_authorization=False,
            run_unchanged=True,
            release_authorization=False,
        )

    def vote(self, principal, body):
        require_role(principal, "offline_reviewer")
        body = validate(OfflineVote, body)
        if not principal.synthetic or body.case_code != self.case_code:
            raise Fault("SCOPE_FORBIDDEN")
        actor = principal.actor_code
        if actor not in (*self.reviewers, self.arbiter):
            raise Fault("SCOPE_FORBIDDEN")
        if body.expected_version != self.subject.checksum:
            raise Fault("VERSION_CONFLICT")
        value = dict(
            schema_version="synthetic-offline-vote/1.0.0",
            actor=actor,
            decision=body.decision,
            request=body.model_dump(mode="json"),
            policy_checksum=digest(self.policy),
        )
        with self.documents.session():
            prior = self.documents.read(self._key(actor))
        if prior:
            if prior[0] != value:
                raise Fault("IDEMPOTENCY_CONFLICT")
            return self.read()
        self._current()
        if actor == self.arbiter and self.read()["status"] != "adjudication_required":
            raise Fault("STATE_CONFLICT")
        with self.documents.session():
            old = self.documents.read(self._key(actor))
            if old and old[0] != value:
                raise Fault("IDEMPOTENCY_CONFLICT")
            self.documents.write(self._key(actor), (value,))
        return self.read()
