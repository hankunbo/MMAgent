"""Exercise frozen 1.0 public writers in tests, never inject JSON into private SQL."""

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path

from db import manage
from packages.grounding.models import validate


@contextmanager
def legacy_discovery(scenario, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    archive = root / "docs/acceptance/11b-input-contract-fix-v1/before"
    name = "_test_11b_legacy"
    with monkeypatch.context() as m:
        spec = importlib.util.spec_from_file_location(
            name,
            archive / "packages/tool_discovery/__init__.py",
            submodule_search_locations=[str(archive / "packages/tool_discovery")],
        )
        package = importlib.util.module_from_spec(spec)
        m.setitem(sys.modules, name, package)
        spec.loader.exec_module(package)
        service = sys.modules[name + ".service"]
        m.setitem(sys.modules, "packages.tool_discovery.service", service)
        spec = importlib.util.spec_from_file_location(
            name + "_repository", archive / "db/runtime/repository/tool_discovery.py"
        )
        repository = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(repository)
        sources = []
        for source in scenario.d.sources:
            records = []
            for record in source._records:
                data = record.model_dump(mode="json")
                data["schema_version"] = "tool-candidate-provenance/1.0.0"
                data.pop("input_mappings")
                data["semantics"].pop("schema_version")
                data["semantics"]["inputs"] = [i["constraint"] for i in data["semantics"]["inputs"]]
                records.append(validate(package.Provenance, data))
            sources.append(
                package.SyntheticSource(
                    source.name,
                    source.kind,
                    records,
                    identities=[i.model_dump(mode="json") for i in source._identities],
                )
            )
        try:
            yield package.Discovery(
                gap_queue=scenario.g,
                repository=repository.DiscoveryRepository(manage.test_target("runtime")),
                sources=tuple(sources),
            )
        finally:
            for key in list(sys.modules):
                if key.startswith(name + "."):
                    del sys.modules[key]
