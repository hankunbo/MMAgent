import pytest

from packages.multimodal_research import alternative_acquisition as a
from packages.multimodal_research import campaign_renewal as c


class OriginalOwnerReached(RuntimeError):
    pass


@pytest.mark.parametrize("method", [a.METHOD, a.TILED_METHOD])
def test_explicit_aggregate_alternative_reaches_original_parent_owner(
    tmp_path, monkeypatch, method
):
    def owner(facts):
        raise OriginalOwnerReached("original current aggregation reader")

    monkeypatch.setattr(c, "completed_parent", owner)
    with pytest.raises(OriginalOwnerReached, match="original current aggregation"):
        c.prepare(
            facts=tmp_path,
            previous=tmp_path,
            output=tmp_path / "new",
            mode="acquisition",
            authorization=(
                "Bounded existing-case alternate observation, original accepted history retained"
            ),
            request_limit=3,
            acquisition_target=["active:rnfl_diffuse"],
            acquisition_increment_only=True,
            alternative_view_method=method,
        )
    assert not (tmp_path / "new").exists()
