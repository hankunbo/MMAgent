"""Original ProcedureReuse boundaries consume the candidate scope per record."""

from types import SimpleNamespace as S

import pytest

from packages.memory.archive import ArchiveSources
from packages.object_store import checksum
from tests.evaluation_preparation.test_multimodal_reuse import fixture


def installed():
    service = fixture()
    state = {"raw": b"original", "reads": 0, "change_on_read": False}

    def metadata(case, code):
        return S(scope=S(case_code=case), checksum=checksum(state["raw"]), research_only=True)

    def read_bytes(asset):
        state["reads"] += 1
        return state["raw"]

    source = ArchiveSources(
        runtime=S(read_case_artifact=metadata),
        read_bytes=read_bytes,
        read_run_case=lambda run: "CASE:OLD",
        read_view=lambda *a: None,
    )
    service.reader.sources = source
    read = service.reader.read

    def with_artifacts(key):
        for _ in range(5):
            source.artifact("CASE:OLD", "A")
        if state["change_on_read"]:
            state["raw"] = b"changed"
        return read(key)

    service.reader.read = with_artifacts
    return service, state


def test_prepare_check_and_cold_restore_each_end_current_proof():
    service, state = installed()
    plan = service.prepare()
    assert state["reads"] == 2
    assert service.check() == ("saved",)
    assert state["reads"] == 4
    assert service.restore(plan) == plan
    assert state["reads"] == 6


@pytest.mark.parametrize("during_prepare", [True, False])
def test_source_change_rejects_before_plan_or_qualified_keys_return(during_prepare):
    service, state = installed()
    if not during_prepare:
        service.prepare()
    state["change_on_read"] = True
    with pytest.raises(ValueError, match="memory_source_changed_during_read"):
        service.prepare() if during_prepare else service.check()
