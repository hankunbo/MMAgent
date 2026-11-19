"""Independent CPU boundaries; no DB, Provider, image inference or PG."""

import copy
import json
from datetime import UTC, datetime, timedelta
from types import MethodType, SimpleNamespace

import pytest

from packages.grounding.models import canonical, digest
from packages.multimodal_research import component_consumer as module
from packages.multimodal_research.campaign import MultimodalCampaign
from packages.multimodal_research.definition_expansion import DefinitionExpansion
from tests.evaluation_preparation.test_component_consumer import config
from tests.evaluation_preparation.test_component_observations import installed_fixture


def guarded_campaign(tmp_path, monkeypatch):
    path, selection = config(tmp_path)
    callbacks = []
    base = SimpleNamespace(
        output=tmp_path,
        observations={},
        active_owner=object(),
    )
    campaign = SimpleNamespace(base=base, output=tmp_path, target="synthetic-db", targets=())
    campaign.guard_window = MethodType(MultimodalCampaign.guard_window, campaign)
    campaign.cumulative_request_limit = MethodType(
        MultimodalCampaign.cumulative_request_limit, campaign
    )
    campaign.guard_control = MethodType(MultimodalCampaign.guard_control, campaign)
    campaign.previous_attempts = lambda: []
    campaign.usage = lambda: 40
    campaign.work_deadline = lambda: datetime.now(UTC) + timedelta(minutes=2)

    def assemble(base, target, selected, *, guard):
        callbacks.append(guard)
        guard()
        return SimpleNamespace(install=lambda: None, restore=lambda: None)

    monkeypatch.setattr(module, "assemble", assemble)
    module.install_campaign(campaign, selection, dict(path=str(path), checksum=digest(selection)))
    campaign.protocol = {"previous_attempts": [], "max_new_requests": 40}
    base.check = callbacks[0]
    return campaign


def test_final_reserved_dispatch_keeps_original_budget_semantics(tmp_path, monkeypatch):
    campaign = guarded_campaign(tmp_path, monkeypatch)
    # A final already-reserved wire is permitted by the existing owner gate.
    campaign.guard_control(dispatch=True)
    # Installing a read-only fact must not silently switch dispatch to False.
    MultimodalCampaign.guard(campaign, dispatch=True)


@pytest.mark.parametrize("stop", ["deadline", "cancel"])
def test_historical_fact_check_does_not_reopen_or_need_live_dispatch(tmp_path, monkeypatch, stop):
    campaign = guarded_campaign(tmp_path, monkeypatch)
    campaign.usage = lambda: 0
    if stop == "deadline":
        campaign.work_deadline = lambda: datetime.now(UTC) - timedelta(seconds=1)
    else:
        (tmp_path / "cancel").write_text("cancelled")
    # Confirm the existing campaign remains blocked for any new work.
    from packages.llm_provider.provider import ProviderError

    with pytest.raises(ProviderError):
        campaign.guard_control(dispatch=True)
    # The adapter callback performs only configuration/current-fact reads.
    campaign.base.check()


def test_zero_increment_revision_keeps_installed_quality_source_type(tmp_path):
    from packages.dataset_adapters.multimodal.adapters import prepare
    from packages.multimodal_research.active_requirements import load, project
    from tests.dataset_adapters.test_multimodal import gamma
    from tests.evaluation_preparation.test_active_acquisition import source_fixture, task

    prepared = prepare("gamma", gamma(tmp_path / "dataset"), "0001")
    rows, _ = project(prepared.manifest, source_fixture(), task(), load())
    target = next(row for row in rows if row["acquisition_method"] == "disc_cup_measurement")
    item = installed_fixture(tmp_path)
    item.target = target["id"]
    item.base.ledger = rows
    item.base.views = {"views": []}
    item.input_row = copy.deepcopy(target)
    item.input_binding = digest((item.input_row, item.input_resources))
    item.install()
    item.check()
    revision = dict(rows=json.loads(canonical(item.base.ledger)), materials=[], added=[])
    expansion = object.__new__(DefinitionExpansion)
    from packages.multimodal_research.definition_expansion import ExpansionPolicy

    expansion.policy = ExpansionPolicy(index_keys=(), public_queries=("optic disc",))
    expansion.campaign = SimpleNamespace(
        base=item.base,
        select_targets=lambda base, available: (),
    )
    expansion.current = lambda: revision
    # Production's exact serialization/reconstruction shape, no medical changes.
    expansion.install(revision)
    item.check()
