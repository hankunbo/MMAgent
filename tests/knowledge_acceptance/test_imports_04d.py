"""Public Knowledge imports must not initialize later-stage agents or model tools."""

import subprocess
import sys
from pathlib import Path


def test_public_knowledge_imports_do_not_load_agents_or_tool_environments():
    script = """
import importlib
import sys
for name in (
    'packages.knowledge_connectors', 'packages.knowledge_search',
    'packages.knowledge_archive', 'packages.knowledge_index',
    'packages.knowledge_claims', 'packages.knowledge_review',
    'packages.retrieval.knowledge', 'packages.disease_packs',
    'db.knowledge.repository', 'apps.knowledge_retrieval',
    'apps.ophthalmology_disease_packs',
):
    importlib.import_module(name)
forbidden = {'agents', 'torch', 'cv2', 'openai', 'transformers',
             'Algorithm_Tool', 'Seg_Model', 'Seg_Agent', 'medgraph'}
assert not forbidden.intersection(n.split('.')[0] for n in sys.modules)
from packages.retrieval import knowledge
assert not hasattr(knowledge, 'SourceCandidateService')
assert 'PublishedKnowledgePort' in knowledge.__all__
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
    )
