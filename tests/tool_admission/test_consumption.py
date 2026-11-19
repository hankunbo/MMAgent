"""Confirmed-source checks are not new execution admission or clinical approval."""

from datetime import timedelta

import pytest

from tests.grounding.query_support import NOW
from tests.tool_admission.support import fixture
from tests.tool_admission.test_research import research


def test_expired_probe_consumption_does_not_authorize_execution(tmp_path):
    f = fixture(tmp_path)
    request = research(f, tmp_path)
    original = f.service.admit_research(request)
    assert original.status == "preparation_only"
    f.service.clock = lambda: NOW + timedelta(days=30)
    assert f.service.validate_research_consumption(request, original) is None
    current = f.service.admit_research(request)
    assert current.status == "rejected"
    assert "probe_stale" in {r.reason for r in current.rejected}
    with pytest.raises(ValueError, match="stale_or_forged_admission_result"):
        f.service.validate_research_result(request, original)


@pytest.mark.parametrize("change", ["withdrawn", "registration", "future_probe"])
def test_consumption_preserves_current_source_gates(tmp_path, change):
    f = fixture(tmp_path)
    request = research(f, tmp_path)
    original = f.service.admit_research(request)
    f.service.clock = lambda: NOW + timedelta(days=30)
    if change == "withdrawn":
        f.service.catalogs.disable(request.candidate.implementation_ref)
        expected = ValueError
    elif change == "registration":
        f.owners.requests.clear()
        expected = KeyError
    else:
        f.service.clock = lambda: NOW - timedelta(days=30)
        expected = ValueError
    with pytest.raises(expected):
        f.service.validate_research_consumption(request, original)
