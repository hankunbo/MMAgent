"""Add per-lease stop observations without modifying the frozen observer/manager."""

from tests.research_task_binding_fix.observer import Observer as PreviousObserver
from tests.research_task_binding_increment.processes import observe_owned
from tests.research_task_binding_r01.observer import export


class Observer(PreviousObserver):
    def pytest_sessionstart(self, session):
        super().pytest_sessionstart(session)
        self.stops = {}
        original_cleanup = self.lease.cleanup

        def cleanup():
            active = self.lease.cluster_id
            if active:
                self.lease._require_owned_state()
            # Existing cleanup verifies cluster ID/marker, pg_ctl stopped and
            # absent postmaster.pid before deleting its own exact directory.
            original_cleanup()
            if active:
                cluster = next(c for c in self.clusters if c["cluster_id"] == active)
                observation = observe_owned(cluster["process"], reap=True)
                value = dict(
                    cluster_id=active,
                    owned_stop_returned=True,
                    root=str(self.lease.root),
                    root_absent=not self.lease.root.exists() and not self.lease.root.is_symlink(),
                    process=observation,
                )
                self.stops[active] = value
                export("stop-" + active, value)

        self.lease.cleanup = cleanup

    def finish(self):
        export("test-reports", self.reports)
        checks = []
        for cluster in self.clusters:
            observation = observe_owned(cluster["process"], reap=True)
            stopped = self.stops.get(cluster["cluster_id"])
            checks.append(
                dict(
                    cluster_id=cluster["cluster_id"],
                    observation=observation,
                    stop_recorded=stopped is not None,
                    stop_verified=bool(
                        stopped
                        and stopped["owned_stop_returned"]
                        and stopped["root_absent"]
                        and stopped["process"]["disposition"] != "running"
                    ),
                )
            )
        history = list(self.lease.created_cluster_ids)
        result = dict(
            database_root=str(self.lease.root),
            session_root=str(self.session_root),
            cluster_ids=history,
            processes=checks,
            database_absent=not self.lease.root.exists() and not self.lease.root.is_symlink(),
            session_absent=not self.session_root.exists() and not self.session_root.is_symlink(),
            cluster_identity_recorded=history == [c["cluster_id"] for c in self.clusters],
        )
        result["verified"] = (
            result["database_absent"]
            and result["session_absent"]
            and result["cluster_identity_recorded"]
            and all(
                c["stop_verified"] and c["observation"]["disposition"] != "running" for c in checks
            )
        )
        export("resources", result)
        return result
