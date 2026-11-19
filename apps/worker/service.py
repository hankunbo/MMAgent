"""Explicit bounded worker: one original Planner command per tick; no implicit daemon."""

from packages.task_runtime.models import TERMINAL
from packages.task_runtime.owner import PlannerTaskOwner


class Cancellation:
    def __init__(self, repository, run_id, external=None):
        self.repository, self.run_id, self.external = repository, run_id, external

    def is_set(self):
        # Read errors propagate; a failed cancellation read never authorizes execution.
        return bool(
            (self.external is not None and self.external.is_set())
            or self.repository.read(self.run_id).cancel_requested
        )


class Worker:
    def __init__(self, repository, *, worker_id):
        if not isinstance(repository.owner, PlannerTaskOwner):
            raise ValueError("dispatch_owner_required")
        self.repository, self.worker_id = repository, worker_id

    def tick(self, run_id, *, claim_key, cancel=None):
        repo = self.repository
        claimed = repo.claim(run_id, claim_key, worker=self.worker_id)
        if claimed.state in TERMINAL:
            return claimed
        token = claimed.lease.token
        params = dict(worker=self.worker_id, token=token)
        with repo.execution_session(run_id):
            repo.check_lease(run_id, **params)
            prepared = repo.command(run_id, "prepare", f"prepare-{token}", **params)
            try:
                repo.check_lease(run_id, **params)
                PlannerTaskOwner.advance(
                    repo.owner, prepared, cancel=Cancellation(repo, run_id, cancel)
                )
            except Exception:
                # No model retry: persist bounded reconciliation, then surface the error.
                # If this write is also unknown, the prepared command remains durable for recovery.
                repo.command(run_id, "failure", f"failure-{token}", **params)
                raise
            return repo.command(run_id, "confirm", f"confirm-{token}", **params)

    def run_bounded(self, run_id, *, batch_key, max_ticks=16, cancel=None, shutdown=None):
        if type(max_ticks) is not int or not 1 <= max_ticks <= 64:
            raise ValueError("dispatch_tick_bound")
        for index in range(max_ticks):
            value = self.repository.read(run_id)
            if value is None or value.state in TERMINAL:
                return value
            if shutdown is not None and shutdown.is_set():
                return value  # worker shutdown is not patient/run cancellation
            value = self.tick(run_id, claim_key=f"{batch_key}-{index}", cancel=cancel)
            if value.state == "confirming":
                return (
                    value  # explicit operator/next bounded invocation, never a tight UNKNOWN loop
                )
        return value
