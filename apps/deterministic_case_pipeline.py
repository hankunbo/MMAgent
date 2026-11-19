"""Explicit 07B visible-only composition. Preparation and evaluation are separate owners."""

from apps.knowledge_retrieval import PublishedKnowledgeService
from apps.ophthalmology_disease_packs import DiseasePackService
from db.audit.repository import PostgresAuditReader, PostgresAuditWriter
from db.runtime.repository.case_pipeline import CasePipelineRepository
from db.runtime.repository.dataset_input import DatasetInputRepository
from db.runtime.repository.grounding import RuntimeGroundingLocks
from packages.dataset_adapters.service import VisibleInputReader
from packages.deterministic_pipeline.demands import DemandResolver
from packages.deterministic_pipeline.service import DeterministicCasePipeline
from packages.deterministic_pipeline.validation import DeterministicValidator
from packages.deterministic_pipeline.validation_ports import assemble_evidence_validator


def assemble_visible_pipeline(
    *,
    runtime_target,
    audit_writer_target,
    audit_reader_target,
    knowledge_reader,
    knowledge_store,
    visible_store,
    catalog,
    configs,
    tools=None,
    mappings=None,
    tool_catalogs=None,
    queries=None,
    evidence_validator_factory=None,
):
    """Targets must be live explicit service identities, never old validation leases.

    Knowledge/Tool owners supply exact configurations and optional formal toolchain.
    No runtime target defaults, catalog discovery, auto-registration or online key changes.
    An optional validator factory is trusted deployment code, never a case request field.
    Installing it does not approve medical rules or upgrade research tool qualification.
    """
    configs = tuple(configs)
    published = PublishedKnowledgeService(knowledge_reader, knowledge_store)
    validator, configs = assemble_evidence_validator(
        DeterministicValidator
        if evidence_validator_factory is None
        else evidence_validator_factory,
        knowledge_reader,
        visible_store,
        published,
        configs,
    )
    runtime = CasePipelineRepository(runtime_target)
    return DeterministicCasePipeline(
        visible_reader=VisibleInputReader(DatasetInputRepository(runtime), visible_store),
        runtime=runtime,
        audit=PostgresAuditWriter(audit_writer_target),
        audit_reader=PostgresAuditReader(audit_reader_target),
        objects=visible_store,
        disease=DiseasePackService(catalog, knowledge_reader, knowledge_store),
        catalog=catalog,
        locks=RuntimeGroundingLocks(runtime_target),
        mappings=mappings,
        tools=tools,
        validator=validator,
        configs=configs,
        demand_resolver=DemandResolver(knowledge=knowledge_reader, catalogs=tool_catalogs),
        queries=queries,
    )
