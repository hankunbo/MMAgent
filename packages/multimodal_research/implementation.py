"""Narrow implementation identity for this composition and its original owners."""

from pathlib import Path

from packages.object_store import checksum


def freeze():
    roots = (
        "packages/multimodal_research",
        "packages/evidence_scoring",
        "packages/evidence_details",
        "packages/evidence_reuse",
        "packages/knowledge_index",
        "packages/llm_provider",
        "packages/research_runtime",
        "packages/dataset_adapters/multimodal",
        "agents/provider_agents",
        "agents/planner",
        "agents/expert_agent",
        "configs/prompts/agents",
    )
    files = {
        p
        for root in roots
        for p in Path(root).rglob("*")
        if p.is_file() and p.suffix in {".py", ".json", ".txt"}
    }
    files.update(
        Path(p)
        for p in (
            "docs/sources/grape-author-2023-PMC10404253.xml",
            "docs/sources/harvard-gdp-author-ICCV2023.pdf",
            "docs/sources/harvard-gdp-author-readme-3d643726.md",
            "packages/evaluation_preparation/dynamic_campaign.py",
            "packages/dataset_adapters/service.py",
            "packages/dataset_adapters/runtime_input.py",
            "packages/dataset_adapters/reading.py",
            "packages/dataset_adapters/prepared.py",
            "packages/dataset_adapters/public_context.py",
            "configs/disease_packs/datasets/fairvlmed_1.0.0.json",
            "configs/disease_packs/fairvlmed_glaucoma_image_note_evidence_v1_1.0.0.json",
            "packages/dataset_adapters/current_read_scope.py",
            "packages/evaluation_preparation/report_production.py",
            "packages/evaluation_preparation/evidence_presentation.py",
            "packages/evaluation_preparation/evidence_tree_static.py",
            "packages/evaluation_preparation/evidence_tree_v3.py",
            "apps/vectorengine_live_assembly.py",
            "packages/planner_control/actions.py",
            "apps/multimodal_case.py",
            "apps/case_memory_store.py",
            "packages/memory/archive.py",
            "packages/memory/archive_objects.py",
            "packages/memory/execution_archive.py",
            "db/memory/repository.py",
            "packages/memory/multimodal_archive.py",
            "packages/memory/procedure_context.py",
            "packages/memory/scope_archive.py",
            "packages/memory/runtime_graph_archive.py",
            "db/runtime/repository/research.py",
            "db/runtime/repository/planner.py",
            "db/runtime/repository/dataset_input.py",
            "db/knowledge/repository/claims.py",
            "db/knowledge/repository/archive.py",
            "db/knowledge/repository/index.py",
            "packages/knowledge_claims/evidence.py",
            "db/knowledge/repository/postgres.py",
            "db/runtime/migrations/0023_task_research_context.sql",
            "configs/research/multimodal_tasks_v1.json",
            "configs/research/multimodal_tasks_v2.json",
            "configs/research/multimodal_tasks_four_hypotheses_v1.json",
            "configs/research/multimodal_propositions_v1.json",
            "configs/research/multimodal_propositions_v2.json",
            "configs/research/requirement_catalog_v1.json",
            "configs/research/requirement_catalog_v2.json",
            "configs/research/requirement_catalog_v3.json",
            "configs/research/multimodal_catalog_acquisition_v1.json",
            "configs/research/multimodal_catalog_acquisition_v2.json",
            "configs/research/multimodal_catalog_acquisition_v3.json",
            "configs/research/multimodal_catalog_acquisition_v4.json",
            "configs/research/multimodal_catalog_acquisition_v5.json",
            "configs/research/multimodal_catalog_acquisition_v6.json",
            "configs/research/multimodal_catalog_acquisition_v7.json",
            "configs/research/multimodal_catalog_acquisition_v8.json",
            "configs/research/multimodal_catalog_acquisition_v9.json",
            "configs/research/input_scope_policy_v1.json",
            "configs/research/input_scope_policy_v2.json",
            "configs/research/input_scope_policy_v3.json",
            "configs/research/input_scope_policy_v4.json",
            "configs/research/multimodal_catalog_acquisition_v10.json",
            "configs/research/input_scope_method_policy_v1.json",
            "configs/research/input_scope_method_policy_v2.json",
            "packages/evidence_acquisition/source_aliases.py",
            "packages/evaluation_preparation/generic_aliases.py",
            "configs/research/multimodal_active_requirements_v1.json",
            "configs/research/multimodal_active_requirements_v2.json",
            "configs/research/multimodal_active_requirements_v3.json",
            "configs/research/multimodal_active_requirements_v4.json",
            "configs/research/multimodal_active_requirements_v5.json",
            "configs/research/multimodal_active_requirements_v6.json",
            "configs/research/multimodal_active_requirements_v7.json",
            "configs/research/multimodal_active_requirements_v8.json",
            "configs/research/multimodal_active_requirements_v9.json",
            "configs/research/multimodal_active_requirements_v10.json",
            "configs/research/multimodal_grape_active_requirements_v1.json",
            "configs/research/multimodal_grape_active_requirements_v2.json",
            "configs/research/multimodal_grape_active_requirements_v3.json",
            "configs/research/multimodal_grape_active_requirements_v4.json",
            "configs/research/multimodal_definition_gate_v1.json",
            "configs/research/multimodal_definition_gate_v2.json",
            "configs/research/multimodal_definition_gate_v3.json",
            "configs/research/multimodal_definition_gate_v4.json",
            "configs/research/multimodal_definition_gate_v5.json",
            "configs/research/multimodal_definition_gate_v6.json",
            "configs/research/multimodal_definition_gate_v7.json",
            "configs/research/multimodal_definition_gate_v8.json",
            "configs/research/multimodal_definition_gate_v9.json",
            "configs/research/multimodal_definition_gate_v10.json",
            "configs/research/multimodal_definition_gate_v11.json",
            "configs/research/multimodal_definition_gate_v12.json",
            "configs/research/multimodal_definition_gate_v13.json",
            "configs/research/multimodal_definition_gate_v14.json",
            "configs/research/multimodal_definition_gate_v15.json",
            "configs/research/multimodal_definition_gate_v16.json",
            "configs/research/multimodal_definition_gate_v17.json",
            "configs/research/multimodal_definition_gate_v18.json",
            "configs/research/multimodal_definition_gate_v19.json",
            "configs/research/multimodal_definition_gate_v20.json",
            "configs/research/multimodal_definition_gate_v21.json",
            "configs/research/multimodal_definition_gate_v21_seed_r1.json",
            "configs/research/multimodal_definition_gate_v22.json",
            "configs/research/multimodal_definition_gap_loop_v1.json",
            "configs/research/multimodal_definition_gap_loop_v2.json",
            "configs/research/multimodal_definition_raw_revision_v1.json",
            "configs/research/multimodal_definition_numeric_interpretation_v1.json",
            "configs/research/multimodal_harvard_visual_priority_v2.json",
            "configs/research/multimodal_harvard_visual_priority_v3.json",
            "configs/research/multimodal_harvard_visual_priority_v4.json",
            "configs/research/multimodal_structural_context_v1.json",
            "configs/research/multimodal_structural_context_v2.json",
            "configs/research/multimodal_structural_coverage_v1.json",
            "configs/research/multimodal_structural_coverage_v2.json",
            "configs/research/multimodal_grape_active_requirements_v5.json",
            "configs/research/multimodal_definition_body_v1.json",
            "packages/knowledge_search/targeted.py",
            "packages/knowledge_search/registered_targets.py",
            "packages/knowledge_search/models.py",
            "packages/knowledge_search/template.py",
            "db/knowledge/migrations/0019_targeted_public_discovery.sql",
            "db/knowledge/migrations/0020_exact_index_material_read.sql",
            "apps/automorph_resources.py",
            "apps/fundus_tool_registry.py",
            "packages/evaluation_preparation/quality.py",
            "apps/disc_geometry.py",
            "packages/tool_execution/fundus_bundle.py",
            "packages/tool_execution/qc.py",
            "packages/tool_execution/service.py",
            "packages/integrations/medgraph/resources.py",
            "packages/tool_admission/service.py",
            "packages/tool_workflow/compiler.py",
            "packages/tool_benchmark/geometry_metrics.py",
            "apps/templates/evidence_tree_reader_v3.js",
            "configs/agents/vectorengine_generic_evidence_v1.json",
            "configs/agents/relayrouter_generic_evidence_v1.json",
            "configs/agents/relayrouter_snapshot_evidence_v1.json",
            "apps/knowledge_literature_tools.py",
            "packages/knowledge_search/executor.py",
            "packages/knowledge_search/policy.py",
        )
    )
    return {str(p): checksum(p.read_bytes()) for p in sorted(files)}


def checked_snapshot(output, original):
    """Keep an original frozen plan plus a separately explicit tested-code amendment.

    This never rewrites the old freeze. Only the narrow, pre-wire search recovery
    owner may create an amendment; every later assembly checks its exact code set.
    """
    current = freeze()
    if (Path(output) / "assembly-repair.json").exists():
        from .assembly_recovery import read_amendment

        value = read_amendment(output)
        if original not in (value["implementation_before"], value["implementation_after"]):
            raise ValueError("multimodal_repair_implementation_changed")
        return original
    if (Path(output) / "phase-budget-repair.json").exists():
        from .context_recovery import read_budget_amendment

        value = read_budget_amendment(output)
        if value["implementation_before"] != original or value["implementation_after"] != current:
            raise ValueError("multimodal_repair_implementation_changed")
        return original
    if (Path(output) / "review-context-repair.json").exists():
        from .context_recovery import read_amendment as read_context_amendment

        value = read_context_amendment(output)
        if value["implementation_before"] != original or value["implementation_after"] != current:
            raise ValueError("multimodal_repair_implementation_changed")
        return original
    path = Path(output) / "local-search-repair.json"
    if not path.exists():
        if original != current:
            raise ValueError("multimodal_entry_implementation_changed")
        return original
    from .search_recovery import read_amendment

    value = read_amendment(output)
    if value["implementation_before"] != original or value["implementation_after"] != current:
        raise ValueError("multimodal_repair_implementation_changed")
    return original  # Historical identity, accompanied by the non-optional amendment.
