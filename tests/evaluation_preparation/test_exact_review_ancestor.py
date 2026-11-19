"""Production reader factory must retain the original route-owning campaign.

These are factory-only tests. Original domain/PG and real read-only proofs are
separate; the document map deliberately models shared Runtime visibility.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from db.runtime.repository.case_pipeline import identity
from packages.grounding.models import digest
from packages.multimodal_research import campaign_renewal as renewal
from tests.evaluation_preparation.test_recovery_network_route import origin
from tests.evaluation_preparation.test_recovery_network_route import route as route


def run_id(name):
    return identity("RUN", "task-research:mm-live:" + digest((name, "review"))[7:55])


def chain(route, descendants, *, client=False):
    old = origin()
    route_plan = dict(
        schema_version=renewal.RETAINED_REVIEW_VERSION,
        mode="review",
        recovery_network_route=route,
    )
    new = renewal.recovery_configuration(old, route_plan)
    if client:
        from packages.llm_provider.client_selection import apply_selection, load_selection

        old = new
        binding = load_selection("configs/agents/system_curl_client_v1.json")
        new = apply_selection(old, binding)
        route_plan = dict(
            schema_version=renewal.RETAINED_ACQUISITION_VERSION,
            mode="acquisition",
            recovery_http_client=binding,
            provider=digest(new),
        )
    saved = {}

    def documents(rid):
        values = saved.get(rid, {})
        return SimpleNamespace(run_id=rid, read=lambda key: values.get(key))

    def node(name, config, parent=None, protocol=None):
        saved[run_id(name)] = {
            "dynamic-phase-input": (dict(config_checksum=digest(config)),),
            "dynamic-phase-result": (
                dict(accepted=True, target=None, acceptance_key="original-accepted"),
            ),
        }
        return SimpleNamespace(
            output=Path(name),
            config=config,
            parent=parent,
            protocol=protocol or dict(schema_version=renewal.RETAINED_ACQUISITION_VERSION),
            documents=documents,
        )

    root = node("original-root", old)
    unaccepted = node("earlier-acquisition", old, root)
    saved[run_id("earlier-acquisition")]["dynamic-phase-result"][0]["accepted"] = False
    routed = node("original-route-review", new, unaccepted, route_plan)
    tip = routed
    for index in range(descendants):
        tip = node(f"later-acquisition-{index}", new, tip)
    return SimpleNamespace(old=old, new=new, root=root, routed=routed, tip=tip, node=node)


@pytest.mark.parametrize("descendants", [0, 1, 2])
def test_cold_factory_keeps_exact_route_owner(route, descendants):
    f = chain(route, descendants)
    rid = run_id(f.tip.output.name)
    reader = renewal.accepted_reader(f.tip, rid)
    if descendants:
        reader = reader.ancestor_reader(run_id(f.routed.output.name), (rid,))
    ancestor = reader.ancestor_reader(run_id(f.root.output.name), (rid,))
    assert ancestor.network_configs == (f.old, f.new)


def test_equal_config_foreign_run_is_not_an_ancestor(route):
    f = chain(route, 1)
    f.node("foreign-same-config", f.new)
    reader = renewal.accepted_reader(f.tip, run_id(f.tip.output.name))
    with pytest.raises(ValueError, match="renewal_accepted_ancestor_unbound"):
        reader.ancestor_reader(run_id("foreign-same-config"), ())


def test_cyclic_parent_chain_cannot_resolve_foreign_run(route):
    f = chain(route, 1)
    f.node("foreign-same-config", f.new)
    f.tip.parent = f.tip
    reader = renewal.accepted_reader(f.tip, run_id(f.tip.output.name))
    with pytest.raises(ValueError, match="renewal_accepted_ancestor_unbound"):
        reader.ancestor_reader(run_id("foreign-same-config"), ())


@pytest.mark.parametrize("descendants", [0, 1, 2])
def test_cold_factory_keeps_exact_client_owner(route, descendants):
    f = chain(route, descendants, client=True)
    rid = run_id(f.tip.output.name)
    reader = renewal.accepted_reader(f.tip, rid)
    if descendants:
        reader = reader.ancestor_reader(run_id(f.routed.output.name), (rid,))
    ancestor = reader.ancestor_reader(run_id(f.root.output.name), (rid,))
    assert ancestor.client_configs == (f.old, f.new)
    assert ancestor.network_configs is None


def test_client_equal_config_foreign_run_is_not_an_ancestor(route):
    f = chain(route, 1, client=True)
    f.node("foreign-same-config", f.new)
    reader = renewal.accepted_reader(f.tip, run_id(f.tip.output.name))
    with pytest.raises(ValueError, match="renewal_accepted_ancestor_unbound"):
        reader.ancestor_reader(run_id("foreign-same-config"), ())
