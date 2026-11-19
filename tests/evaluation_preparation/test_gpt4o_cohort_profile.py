"""The resumed cohort uses existing routed owners, not the old alternate binding."""

from pathlib import Path

import pytest

from packages.evaluation_preparation import terminal_cohort as owner
from packages.evaluation_preparation.generic_input import ROUTED_PROVIDER
from packages.llm_provider.config import read_provider_config
from packages.llm_provider.network_route import load_route
from packages.multimodal_research.campaign import configuration


@pytest.mark.parametrize(
    ("name", "harvard"),
    [
        ("terminal_cohort_gpt4o_coverage_v1.json", "data_0012.npz"),
        ("terminal_cohort_gpt4o_fresh_gate_v1.json", "data_0020.npz"),
    ],
)
def test_gpt4o_cohort_resolves_original_production_providers_and_exact_commands(
    tmp_path, name, harvard
):
    manifest = owner.read_manifest(owner.read("configs/research/" + name))
    assert manifest.schema_version == "terminal-case-cohort/1.3.0"
    assert not isinstance(manifest, owner.ModelSelectedManifest)
    route = load_route(manifest.network_route)
    for case in manifest.cases:
        argv = owner.commands(manifest, case, tmp_path, "prepare")[0]
        assert "--model-selection" not in argv
        if case.dataset == "fairvlmed":
            assert argv[argv.index("--provider-profile") + 1] == "relay-grounded-shape-proxy-v1"
            config = read_provider_config(owner.read(ROUTED_PROVIDER))
        else:
            assert argv[argv.index("--execution-profile") + 1] == "relay-root-question-v1"
            config = configuration(
                resilient_vision=True,
                directional=True,
                exact_primary_status=True,
                qualified_observations=bool(case.input_semantics),
                execution_profile="relay-root-question-v1",
                review_batch_size=case.review_batch_size,
                network_route=route.model_dump(mode="json"),
            )
        assert config.transport.model == "gpt-4o-2024-08-06"
        assert config.transport.base_url == "https://api.relayrouter.ai/v1"
        assert config.network_route == route
        assert not any(getattr(role, "model", None) for role in config.roles)
    assert {c.dataset for c in manifest.cases} == {"fairvlmed", "gamma", "harvard_gdp", "grape"}
    assert next(c.sample for c in manifest.cases if c.dataset == "harvard_gdp") == harvard
    # The old gpt-6 selection and failed G cohort remain unchanged and readable.
    old = owner.read_manifest(
        owner.read("configs/research/terminal_cohort_alternate_coverage_v1.json")
    )
    assert old.schema_version == "terminal-case-cohort/1.4.0"
    assert Path(old.model_selection).is_file()
