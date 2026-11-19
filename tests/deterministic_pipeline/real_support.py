"""Visible-only application assembly shared by synthetic data tests and explicit real probe."""

import json
from pathlib import Path

from apps.knowledge_retrieval import PublishedKnowledgeService
from apps.ophthalmology_disease_packs import DiseasePackService
from db import manage
from db.audit.repository import PostgresAuditReader, PostgresAuditWriter
from db.runtime.repository.case_pipeline import CasePipelineRepository
from db.runtime.repository.dataset_input import DatasetInputRepository
from packages.dataset_adapters.service import VisibleInputReader
from packages.deterministic_pipeline.models import PipelineConfig
from packages.deterministic_pipeline.service import DeterministicCasePipeline
from packages.deterministic_pipeline.validation import DeterministicValidator
from packages.disease_packs import DiseasePackCatalog

ROOT = Path("/opt/data/test/MMAgent")


def assembly(database, visible):
    rt = manage.test_target("runtime")
    manage.bootstrap_runtime_principal(database.admin, rt.user)
    manage.bootstrap_audit_principals(
        database.admin,
        manage.test_target("audit_writer").user,
        manage.test_target("audit_reader").user,
    )
    runtime = CasePipelineRepository(rt)
    catalog = DiseasePackCatalog.from_directory(ROOT / "configs/disease_packs")
    published = PublishedKnowledgeService(database.reader, visible)
    validator = DeterministicValidator(database.reader, visible, published)
    cfg = PipelineConfig.model_validate_json(
        (ROOT / "configs/deterministic_pipeline/fairvlmed_gap.json").read_bytes()
    )
    service = DeterministicCasePipeline(
        visible_reader=VisibleInputReader(DatasetInputRepository(runtime), visible),
        runtime=runtime,
        audit=PostgresAuditWriter(manage.test_target("audit_writer")),
        audit_reader=PostgresAuditReader(manage.test_target("audit_reader")),
        objects=visible,
        disease=DiseasePackService(catalog, database.reader, visible),
        catalog=catalog,
        locks=None,
        mappings=None,
        tools=None,
        validator=validator,
        configs=(cfg,),
    )
    return service, cfg


def write_evidence(service, report, directory):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    traces = [
        service._read(
            __import__(
                "packages.dataset_adapters.models", fromlist=["ObjectReference"]
            ).ObjectReference.model_validate(r)
        )
        for r in report["trace"]
    ]
    (directory / "trace.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n")
