"""New collector observer: multiple owned leases, fixed inputs, failure diagnostics."""

import os
from copy import deepcopy
from pathlib import Path

import pytest

from db import manage
from db.runtime.repository.research_lifecycle import ResearchLifecycleRepository
from packages.grounding.models import digest
from packages.research_runtime.single_case import ResearchSingleCaseService
from tests.research_task_binding_fix.support import fixed_input
from tests.research_task_binding_r01.observer import Observer as OriginalObserver
from tests.research_task_binding_r01.observer import export, process_identity
from tests.research_task_binding_r01.test_postgres import owner_snapshot
from tests.research_task_completion import support


class Observer(OriginalObserver):
    def pytest_sessionstart(self, session):
        super().pytest_sessionstart(session)
        self.clusters = []
        self.produced = {}
        self.returned = {}
        patch = pytest.MonkeyPatch()
        session.config.add_cleanup(patch.undo)
        patch.setattr(support, "synthetic_input", fixed_input)
        original_start = self.lease.start

        def start(port):
            result = original_start(port)
            self.lease._require_owned_state()
            state = manage._read_test_state()
            data = Path(state["data_dir"])
            pid = int((data / "postmaster.pid").read_text().splitlines()[0])
            process = process_identity(pid)
            assert process and process["name"] == "postgres"
            assert manage._pg_ctl_status(data).returncode == 0
            doc = dict(
                root=str(self.lease.root),
                cluster_id=result,
                data=str(data),
                process=process,
                ownership_verified=True,
            )
            self.clusters.append(doc)
            export("cluster-" + result, doc)
            return result

        patch.setattr(self.lease, "start", start)
        original_record = ResearchLifecycleRepository.record_lifecycle

        def record(runtime, old, **kwargs):
            if kwargs.get("report") is not None:
                self.produced[old["run_id"]] = deepcopy(kwargs["report"])
            return original_record(runtime, old, **kwargs)

        patch.setattr(ResearchLifecycleRepository, "record_lifecycle", record)
        original_run = ResearchSingleCaseService.run_registered

        def run(driver, request, **kwargs):
            report = original_run(driver, request, **kwargs)
            self.returned[request.lock.run_id] = deepcopy(report)
            return report

        patch.setattr(ResearchSingleCaseService, "run_registered", run)
        original_read = ResearchSingleCaseService.read_task_completion

        def read(driver, rid):
            try:
                return original_read(driver, rid)
            except BaseException as error:
                # Includes module fixture preparation, before its owned PG cleanup.
                if str(error) == "task_completion_report_binding":
                    name = "binding-rejection-" + digest(rid)[7:]
                    directory = Path(os.environ["MMAGENT_BINDING_DIAGNOSTIC_DIR"])
                    if not (directory / (name + ".json")).exists():
                        doc = dict(error_type=type(error).__name__, public_error=str(error))
                        try:
                            state = driver.runtime.lifecycle(rid)
                            context = driver.research.read(rid)
                            choice = (
                                driver.choice(context.request)
                                if callable(driver.choice)
                                else driver.choice
                            )
                            plan = driver.research.compile(rid, choice)
                            key = state["binding"]["execution_key"]
                            services = {key: driver.bind_plan(plan, key)}
                            doc["snapshot"] = owner_snapshot(
                                driver,
                                context.request,
                                self.returned.get(rid),
                                self.produced.get(rid),
                                services,
                                key.removesuffix(":tool"),
                            )
                        except Exception as diagnostic_error:
                            doc["diagnostic_error_type"] = type(diagnostic_error).__name__
                        try:
                            export(name, doc)
                        except Exception as export_error:
                            error.add_note(
                                "diagnostic_export_failed:" + type(export_error).__name__
                            )
                raise

        patch.setattr(ResearchSingleCaseService, "read_task_completion", read)

    def finish(self):
        export("test-reports", self.reports)
        checks = []
        for cluster in self.clusters:
            current = process_identity(cluster["process"]["pid"])
            alive = bool(current and current["start_ticks"] == cluster["process"]["start_ticks"])
            checks.append(dict(cluster_id=cluster["cluster_id"], original_process_alive=alive))
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
            and all(not c["original_process_alive"] for c in checks)
        )
        export("resources", result)
        return result
