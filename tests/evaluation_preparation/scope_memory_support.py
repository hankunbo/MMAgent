"""Original Memory repositories on the same isolated test database."""

from db import manage
from db.memory.admin import bootstrap_memory_principals
from db.memory.repository import MemoryRepository
from packages.memory.archive import ArchiveSources
from packages.multimodal_research import memory, report


def install(database, campaign, tmp_path, monkeypatch):
    bootstrap_memory_principals(
        database[0], *(manage.test_target(c).user for c in manage.MEMORY_CHANNELS)
    )
    # The final run exists before this helper; all dependencies are original owners.
    final = campaign.phase_path(None) / "result.json"
    import json

    documents = campaign.documents(json.loads(final.read_bytes())["run_id"])
    owner = ArchiveSources(
        runtime=campaign.base.runtime,
        read_bytes=documents.read_artifact,
        read_run_case=lambda run: campaign.contexts.read_research_run(run).request.scope.case_code,
        read_view=lambda *_: None,
    )
    writer = MemoryRepository(
        manage.test_target("memory_writer"), capability="writer", sources=owner
    )
    monkeypatch.setattr(memory, "archive", lambda c, r: memory.persist(writer, owner, documents, r))
    original_export, original_restore = report.export, report.restore_export
    monkeypatch.setattr(
        report, "export", lambda c, r: original_export(c, r, directory=tmp_path / "exports")
    )
    monkeypatch.setattr(
        report,
        "restore_export",
        lambda c, r: original_restore(c, r, directory=tmp_path / "exports"),
    )
    return owner, writer
