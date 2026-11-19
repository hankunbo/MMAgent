"""Public import isolation and repository trust boundaries."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from packages.grounding import GroundingError, Mapping, Policy, verify
from packages.grounding.synthetic import SyntheticCatalogReader, fixture
from packages.retrieval.knowledge.models import RetrievalContext
from tests.grounding.test_models import reseal

ROOT = Path(__file__).resolve().parents[2]


def test_public_import_is_offline_and_has_no_private_domain_dependencies():
    script = """
import sys
import packages.grounding
blocked = {'db','apps','agents','openai','torch','transformers','psycopg'}
assert not any(n.split('.')[0] in blocked for n in sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for source in (ROOT / "db/grounding").rglob("*.py"):
        parsed = ast.parse(source.read_text())
        imports = [n.module or "" for n in ast.walk(parsed) if isinstance(n, ast.ImportFrom)]
        assert not any("knowledge.repository." in n or "runtime.repository." in n for n in imports)
        assert not any("medgraph" in n or n.startswith(("apps", "agents")) for n in imports)


@pytest.mark.parametrize("status", ["registered", "quarantined", "failed", "deprecated"])
def test_catalog_status_cannot_self_authorize(status):
    m, p, c, k, _ = fixture()
    data = c.model_dump(mode="json")
    data["tools"][0]["catalog_status"] = status
    new = reseal(type(c), data)
    m = reseal(
        Mapping,
        {**m.model_dump(mode="json"), "catalog_ref": new.reference().model_dump(mode="json")},
    )
    with pytest.raises(GroundingError, match="tool_not_active"):
        verify(m, p, k, SyntheticCatalogReader([new]), RetrievalContext(), admission=True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("admissible_states", ["Draft"]),
        ("type_compatibility", "arbitrary"),
        ("max_mappings", 10000),
    ],
)
def test_policy_cannot_bypass_enums_and_scope(field, value):
    _, p, _, _, _ = fixture()
    with pytest.raises(ValueError):
        reseal(Policy, {**p.model_dump(mode="json"), field: value})
