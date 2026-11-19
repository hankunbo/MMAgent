from types import SimpleNamespace

import pytest

from db.knowledge import models as m
from db.knowledge.repository.archive import _bounded_confirmed


@pytest.mark.parametrize("reason", ["page_limit", "hit_budget"])
def test_only_confirmed_bounded_pages(reason):
    outcome = dict(
        stop_reason=reason,
        error_category=None,
        error_requests=(),
        hit_count=1,
        receipts=(object(),),
        wire_receipts=(SimpleNamespace(http_status=200, response_checksum="sha256:real"),),
    )
    run = SimpleNamespace(run_status=m.JobStatus.FAILED, execution=SimpleNamespace(**outcome))
    assert _bounded_confirmed(run)
    for field, invalid in (
        ("stop_reason", "timeout"),
        ("stop_reason", "cancelled"),
        ("stop_reason", "duplicate_page"),
        ("stop_reason", "request_budget"),
        ("error_category", "network"),
        ("error_requests", (object(),)),
        ("hit_count", 0),
        ("receipts", ()),
        ("wire_receipts", ()),
        ("wire_receipts", (SimpleNamespace(http_status=None, response_checksum=None),)),
        ("wire_receipts", (SimpleNamespace(http_status=500, response_checksum="sha256:error"),)),
    ):
        run.execution = SimpleNamespace(**(outcome | {field: invalid}))
        assert not _bounded_confirmed(run), field
