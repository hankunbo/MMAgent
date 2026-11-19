"""Fresh 05/06 checks through installed owners; no tool execution or new run creation."""

from db.runtime.repository.grounding_query import GroundingQueryService
from packages.grounding.models import digest
from packages.tool_admission.service import ToolAdmissionService

from .catalog import SyntheticCatalogOwner


class SyntheticFutureRunOwner:
    def __init__(
        self,
        *,
        catalog,
        admission,
        contexts=None,
        run_id=None,
        query_reader=None,
        selection_reader=None,
    ):
        if type(catalog) is not SyntheticCatalogOwner or not isinstance(
            admission, ToolAdmissionService
        ):
            raise ValueError("extension_future_owner_required")
        if admission.catalogs is not catalog.reader:
            raise ValueError("extension_future_catalog_owner_mismatch")
        self.catalog, self.admission, self.contexts, self.run_id = (
            catalog,
            admission,
            contexts,
            run_id,
        )
        self.query_reader, self.selection_reader = query_reader, selection_reader

    def binding(self, original):
        if not self.run_id or self.run_id == original.application.run_id:
            raise ValueError("extension_explicit_new_run_required")
        origin_scope = original.application.material.get("request", {}).get("scope")
        if origin_scope is None:
            raise ValueError("extension_original_scope_owner_not_installed")
        if self.query_reader is not None:
            if self.selection_reader is None or not isinstance(
                self.admission.grounding, GroundingQueryService
            ):
                raise ValueError("extension_original_05_06_required")
            q = self.query_reader(self.run_id)
            if not q.lock.policy.synthetic_only:
                raise ValueError("extension_synthetic_query_policy_required")
            selection = self.selection_reader(self.run_id)
            if q.lock.run_id != self.run_id or q.scope.model_dump(mode="json") != origin_scope:
                raise ValueError("extension_future_scope")
            if digest(q.lock.policy) != self.catalog.fixture.fixture_policy_checksum:
                raise ValueError("extension_fixture_policy_mismatch")
            catalog = q.lock.catalog_ref
            material = dict(
                mode="05_06",
                request=q.model_dump(mode="json"),
                selection=selection.model_dump(mode="json"),
            )
        else:
            if self.contexts is None:
                raise ValueError("extension_future_runtime_required")
            ctx = self.contexts.read_research_run(self.run_id)
            if not ctx.synthetic or ctx.request.scope.model_dump(mode="json") != origin_scope:
                raise ValueError("extension_future_scope")
            if ctx.request.candidate.implementation_ref != self.catalog.fixture.implementation_ref:
                raise ValueError("extension_future_implementation_mismatch")
            if (
                digest(ctx.request.compatibility_policy)
                != self.catalog.fixture.fixture_policy_checksum
            ):
                raise ValueError("extension_fixture_policy_mismatch")
            catalog = ctx.request.lock.catalog_ref
            material = dict(mode="06_research", context=ctx.model_dump(mode="json"))
        if (
            catalog != self.catalog.fixture.catalog_ref
            or catalog.model_dump(mode="json") == original.application.catalog
        ):
            raise ValueError("extension_new_catalog_lock_required")
        return dict(new_run_id=self.run_id, catalog_ref=catalog.model_dump(mode="json"), **material)

    def _recheck(self, plan):
        binding = self.binding(plan.adapter_plan)
        if binding != plan.future_binding:
            raise ValueError("extension_future_binding_changed")
        if self.catalog.current(plan.adapter_record) != plan.catalog_observation:
            raise ValueError("extension_current_catalog_changed")
        return binding

    def evaluate(self, plan):
        binding = self._recheck(plan)
        if binding["mode"] == "05_06":
            q = self.query_reader(self.run_id)
            query = self.admission.grounding.query(q)
            admission = self.admission.admit(q, query, self.selection_reader(self.run_id))
            query_doc = query.model_dump(mode="json")
        else:
            request = self.contexts.read_research_run(self.run_id).request
            admission = self.admission.admit_research(request)
            query_doc = None
        # Both original admission paths return before current input/selection and
        # catalog qualification are checked again. Never rerun admission here.
        self._recheck(plan)
        retained = any(
            c.candidate.implementation_ref == self.catalog.fixture.implementation_ref
            for c in admission.retained
        )
        # A complete result is evidence only, not an ExecutionPlan or a replay success.
        doc = dict(
            schema_version="synthetic-extension-future-check/1.0.0",
            synthetic=True,
            original_run_id=plan.grant.run_id,
            original_lock=plan.adapter_plan.application.catalog,
            original_application=plan.adapter_plan.application.model_dump(mode="json"),
            original_runtime_context=plan.runtime_context,
            new_run_id=self.run_id,
            new_lock=binding["catalog_ref"],
            binding=binding,
            grounding=query_doc,
            grounding_status="recomputed" if query_doc else "not_applicable_registered_research",
            admission=admission.model_dump(mode="json"),
            fixture_implementation_retained=retained,
            execution="not_executed",
            counterfactual_status="prerequisites_only_no_executed_pair",
            memory_published=False,
            available_next_run=False,
        )
        return dict(**doc, evidence_checksum=digest(doc))
