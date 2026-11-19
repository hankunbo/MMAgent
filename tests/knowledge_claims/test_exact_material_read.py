"""Optional exact-index port preserves old deployments and never masks a rejection."""

from types import SimpleNamespace

import pytest

from db.knowledge.repository import claims
from packages.knowledge_index.models import Filters


class Connection:
    def __init__(self, installed, failure=None):
        self.installed, self.failure, self.calls = installed, failure, []

    def execute(self, statement, arguments=()):
        self.calls.append((statement, arguments))
        if "to_regprocedure" in statement:
            return SimpleNamespace(fetchone=lambda: (self.installed,))
        if self.failure:
            raise self.failure
        return SimpleNamespace(fetchone=lambda: ({"entries": []},))


@pytest.mark.parametrize("installed", [False, True])
def test_exact_current_port_or_original_fallback(installed):
    connection = Connection(installed)
    hit = {"index_key": "index", "chunk_ref": {"code": "SYNTHETIC:CHUNK"}}
    assert claims._read_current_index(connection, hit, Filters()) == {"entries": []}
    assert len(connection.calls) == 2
    statement, arguments = connection.calls[-1]
    if installed:
        assert "knowledge.read_document_index_entry(" in statement
        assert arguments[1].obj == hit["chunk_ref"]
    else:
        assert "knowledge.read_document_index(" in statement
    assert arguments[0] == hit["index_key"]


def test_installed_current_port_error_cannot_fall_back():
    failure = RuntimeError("synthetic_current_source_failure")
    connection = Connection(True, failure)
    with pytest.raises(RuntimeError, match="synthetic_current_source_failure"):
        claims._read_current_index(connection, {"index_key": "i", "chunk_ref": {}}, Filters())
    assert len(connection.calls) == 2
    assert "read_document_index_entry(" in connection.calls[-1][0]


def test_capability_is_rechecked_not_cached_between_calls():
    connection = Connection(False)
    hit = {"index_key": "i", "chunk_ref": {}}
    claims._read_current_index(connection, hit, Filters())
    connection.installed = True
    claims._read_current_index(connection, hit, Filters())
    assert len(connection.calls) == 4
    assert "read_document_index_entry(" in connection.calls[-1][0]
