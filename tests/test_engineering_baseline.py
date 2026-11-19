"""Engineering-boundary acceptance tests through stage 01B."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV001_PYTHON = Path("/opt/data/test/envs/MMAgent1/bin/python").resolve()

REQUIRED_READMES = (
    "apps/api/README.md",
    "apps/worker/README.md",
    "agents/planner/README.md",
    "agents/trace_agent/README.md",
    "agents/expert_agent/README.md",
    "agents/tool_agent/README.md",
    "agents/verification_agent/README.md",
    "packages/contracts/README.md",
    "packages/graph_engine/README.md",
    "packages/rule_engine/README.md",
    "packages/retrieval/knowledge/README.md",
    "packages/retrieval/memory/README.md",
    "packages/workflow_compiler/README.md",
    "packages/memory_writer/README.md",
    "packages/object_store/README.md",
    "packages/provenance/README.md",
    "db/core/README.md",
    "db/knowledge/README.md",
    "db/grounding/README.md",
    "db/runtime/README.md",
    "db/memory/README.md",
    "db/audit/README.md",
    "configs/prompts/README.md",
    "configs/policies/README.md",
    "tests/contracts/README.md",
    "tests/graph/README.md",
    "tests/tools/README.md",
    "tests/integration/README.md",
    "tests/golden_cases/README.md",
    "docs/architecture_boundaries.md",
)


def test_env001_is_the_project_interpreter() -> None:
    assert Path(sys.executable).resolve() == ENV001_PYTHON


def test_required_boundaries_are_documented() -> None:
    for relative_path in REQUIRED_READMES:
        document = ROOT / relative_path
        assert document.is_file(), relative_path
        text = document.read_text(encoding="utf-8")
        if document.name == "README.md":
            assert "允许依赖" in text, relative_path
            assert "禁止依赖" in text, relative_path


def test_existing_tool_assets_remain_in_place() -> None:
    for relative_path in (
        "algorithm-capability-db",
        "Algorithm_Tool",
        "Seg_Model",
        "Seg_Agent",
    ):
        assert (ROOT / relative_path).is_dir(), relative_path


def test_step01_contract_public_entry_has_only_approved_runtime_dependencies() -> None:
    contract_entry = ROOT / "packages/contracts/__init__.py"
    assert contract_entry.is_file()

    forbidden_roots = {
        "Algorithm_Tool",
        "Seg_Model",
        "agents",
        "apps",
        "cv2",
        "db",
        "openai",
        "psycopg",
        "sqlalchemy",
        "torch",
    }
    imported_roots: set[str] = set()
    for source_path in (ROOT / "packages/contracts").glob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.partition(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported_roots.add(node.module.partition(".")[0])
    assert not (forbidden_roots & imported_roots)

    script = """
import sys
import packages.contracts

forbidden_roots = {
    "agents", "apps", "db", "openai", "sqlalchemy", "psycopg",
    "cv2", "torch", "Algorithm_Tool", "Seg_Model",
}
loaded_roots = {name.partition(".")[0] for name in sys.modules}
unexpected = sorted(forbidden_roots & loaded_roots)
assert not unexpected, unexpected
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_public_env_template_does_not_contain_a_key_value() -> None:
    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=" in env_example
    key_line = next(line for line in env_example.splitlines() if line.startswith("OPENAI_API_KEY="))
    assert key_line == "OPENAI_API_KEY="


def test_sensitive_and_large_paths_are_git_ignored(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text(
        (ROOT / ".gitignore").read_text(encoding="utf-8"), encoding="utf-8"
    )
    subprocess.run(
        ["git", "init", "--quiet"], cwd=tmp_path, check=True, capture_output=True, text=True
    )
    candidates = (
        ".env",
        "service/.env.local",
        "algorithm-capability-db/.postgres/data/PG_VERSION",
        "algorithm-capability-db/mcp_server/.venv/bin/python",
        "Seg_Model/example/weights/model.pth",
        "Seg_Model/test/T0001_optic_disc_mask.png",
        "Seg_Model/example/source_medagent_pro",
        "cases/patient.npz",
        "runs/RUN_001/result.json",
    )
    for candidate in candidates:
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", "--no-index", candidate],
            cwd=tmp_path,
            check=False,
        )
        assert result.returncode == 0, candidate

    public_template = subprocess.run(
        ["git", "check-ignore", "--quiet", "--no-index", ".env.example"],
        cwd=tmp_path,
        check=False,
    )
    assert public_template.returncode == 1


def test_env001_is_registered() -> None:
    registry = (
        ROOT / "Total_graph_step_planner/详细实施版/共享规范/08_全流程技术决策登记表.md"
    ).read_text(encoding="utf-8")
    assert "| ENV-001 |" in registry
    assert "/opt/data/test/envs/MMAgent1" in registry
