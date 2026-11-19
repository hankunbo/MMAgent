"""Protocol-kind checks, separate from original medical owner qualification."""

from types import SimpleNamespace as NS

import pytest

from packages.multimodal_research import alternative_acquisition as a


def plan(version, method):
    return dict(
        schema_version=version,
        mode="acquisition",
        seconds=5400,
        acquisition_targets=["check"],
        max_new_requests=3,
        new_public_search_allowed=False,
        alternative_view={"method": method},
        acquisition_increment_only=version == a.AGGREGATE_VERSION,
    )


@pytest.mark.parametrize(
    "version,method", [(a.VERSION, a.METHOD), (a.TILED_VERSION, a.TILED_METHOD)]
)
def test_legacy_alternative_cannot_consume_aggregate_by_implicit_upgrade(
    monkeypatch, version, method
):
    monkeypatch.setattr(a, "proof", lambda *_args, **_kwargs: {"method": method})
    with pytest.raises(ValueError, match="alternative_view_aggregation_plan_binding"):
        a.check(NS(aggregation_baseline=object()), plan(version, method))


@pytest.mark.parametrize("method", [a.METHOD, a.TILED_METHOD])
def test_new_explicit_scope_requires_aggregate_but_preserves_legacy_read(monkeypatch, method):
    monkeypatch.setattr(a, "proof", lambda *_args, **_kwargs: {"method": method})
    assert a.check(NS(aggregation_baseline=object()), plan(a.AGGREGATE_VERSION, method)) == [
        "check"
    ]
    with pytest.raises(ValueError, match="alternative_view_aggregation_plan_binding"):
        a.check(NS(), plan(a.AGGREGATE_VERSION, method))
    assert a.check(NS(), plan(a.METHOD_VERSIONS[method], method)) == ["check"]
