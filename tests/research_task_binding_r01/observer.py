"""Structured ownership capture; no log-path parsing or independent deletion authority."""

import json
import os
from pathlib import Path

from db import manage


def export(name, document):
    root = Path(os.environ["MMAGENT_BINDING_DIAGNOSTIC_DIR"])
    with (root / (name + ".json")).open("x") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def process_identity(pid):
    try:
        # No cmdline/environment export: only PID start time and the executable name.
        raw = Path(f"/proc/{pid}/stat").read_text()
        fields = raw[raw.rindex(")") + 2 :].split()
        return dict(
            pid=pid, start_ticks=fields[19], name=Path(f"/proc/{pid}/comm").read_text().strip()
        )
    except FileNotFoundError:
        return None


def record_cluster(lease):
    lease._require_owned_state()
    state = manage._read_test_state()
    data = Path(state["data_dir"])
    pid = int((data / "postmaster.pid").read_text().splitlines()[0])
    process = process_identity(pid)
    assert process is not None and process["name"] == "postgres"
    assert manage._pg_ctl_status(data).returncode == 0
    value = dict(
        root=str(lease.root),
        cluster_id=lease.cluster_id,
        data=str(data),
        socket=state["socket_dir"],
        process=process,
        ownership_verified=True,
    )
    export("cluster-start", value)


class Observer:
    def __init__(self):
        self.reports = []

    def pytest_collection_finish(self, session):
        export("collected", dict(nodeids=[item.nodeid for item in session.items]))

    def pytest_runtest_logreport(self, report):
        self.reports.append(dict(nodeid=report.nodeid, when=report.when, outcome=report.outcome))

    def pytest_sessionstart(self, session):
        from tests.database.pytest_plugin import _LEASE

        self.lease = session.config.stash[_LEASE]
        self.session_root = Path(session.config.option.basetemp).parent
        self.temp_owner = json.loads((self.session_root / "owner.json").read_text())
        self.inode = self.session_root.stat().st_ino
        assert self.temp_owner["pid"] == os.getpid()
        export(
            "session-start",
            dict(
                database_root=str(self.lease.root),
                session_root=str(self.session_root),
                session_inode=self.inode,
                session_owner=self.temp_owner,
            ),
        )

    def finish(self):
        export("test-reports", self.reports)
        root = Path(os.environ["MMAGENT_BINDING_DIAGNOSTIC_DIR"])
        cluster = root / "cluster-start.json"
        started = json.loads(cluster.read_text()) if cluster.exists() else None
        remaining = process_identity(started["process"]["pid"]) if started else None
        same_process = bool(
            started and remaining and remaining["start_ticks"] == started["process"]["start_ticks"]
        )
        history = list(self.lease.created_cluster_ids)
        value = dict(
            database_root=str(self.lease.root),
            session_root=str(self.session_root),
            cluster_ids=history,
            database_absent=not self.lease.root.exists() and not self.lease.root.is_symlink(),
            session_absent=not self.session_root.exists() and not self.session_root.is_symlink(),
            owned_process_alive=same_process,
            cluster_identity_recorded=not history
            or bool(started and history == [started["cluster_id"]]),
        )
        value["verified"] = (
            value["database_absent"]
            and value["session_absent"]
            and not same_process
            and value["cluster_identity_recorded"]
        )
        export("resources", value)
        return value
