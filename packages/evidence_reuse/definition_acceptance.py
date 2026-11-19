"""Research-only definition owner over original RuntimeDocuments, never publication.

The trusted source reader reloads original Knowledge evidence. No caller supplied
validated bit is accepted. Text entailment remains a bounded research judgment.
"""

from copy import deepcopy

from packages.grounding.models import canonical, digest, validate
from packages.llm_provider.models import Source, TaskDefinition

from .definitions import (
    ConfirmedInterpretationDraft,
    EvidenceResearchDefinitionDraft,
    OwnerBoundConfirmedInterpretationDraft,
    OwnerBoundResearchDefinitionDraft,
    ResearchDefinitionDraft,
    ScopedResearchDefinitionDraft,
    locate_draft,
)

CONFIRMED_RECEIPTS = {"accepted-research-definition/4.0.0", "accepted-research-definition/6.0.0"}
OWNER_BOUND_RECEIPTS = {"accepted-research-definition/5.0.0", "accepted-research-definition/6.0.0"}


class _BuildSources:
    """One reconstruction, bracketed by fresh reads from the original owner.

    Only repeated quote parsing uses this local snapshot. It cannot authorize a
    later build, request or commit; verify() reloads every source before return.
    """

    def __init__(self, owner):
        self.owner, self.values = owner, {}

    def read(self, code):
        if code not in self.values:
            self.values[code] = deepcopy(self.owner.read(code))
        return deepcopy(self.values[code])

    def verify(self):
        for code, value in self.values.items():
            if self.owner.read(code) != value:
                raise ValueError("definition_draft_source_changed")


class ResearchDefinitionOwner:
    def __init__(
        self,
        *,
        documents,
        sources,
        campaign_id,
        target,
        input_checksum,
        guard,
        input_reader=None,
        fact_reader=None,
        confirmed_observations=None,
        requirement_ids=(),
    ):
        self.documents, self.sources, self.guard = documents, sources, guard
        self.input_reader = input_reader
        self.fact_reader = fact_reader
        self.confirmed_observations = confirmed_observations
        self.requirement_ids = tuple(requirement_ids)
        if len(set(self.requirement_ids)) != len(self.requirement_ids):
            raise ValueError("definition_confirmed_requirement_aliases")
        self.binding = dict(campaign_id=campaign_id, target=target, input_checksum=input_checksum)

    def build(self, draft):
        self.guard()
        return self._build(draft)

    def _build(self, draft):
        """Source/content reconstruction; no write or dispatch authority."""
        candidate = draft.model_dump(mode="json") if hasattr(draft, "model_dump") else draft
        located = candidate.get("schema_version") in {
            "source-definition-draft/2.1.0",
            "source-definition-draft/3.1.0",
            "source-definition-draft/4.1.0",
            "source-definition-draft/5.1.0",
            "source-definition-draft/6.1.0",
            "source-definition-draft/7.1.0",
        }
        source_read = _BuildSources(self.sources)
        draft = locate_draft(candidate, target=self.binding["target"], source_owner=source_read)
        confirmed = (
            self.confirmed_basis(draft) if isinstance(draft, ConfirmedInterpretationDraft) else None
        )
        assessment = None
        typed = isinstance(
            draft, (OwnerBoundResearchDefinitionDraft, OwnerBoundConfirmedInterpretationDraft)
        )
        typed_manifest = None
        if isinstance(draft, ScopedResearchDefinitionDraft):
            from .definition_dependencies import assess

            if self.input_reader is None:
                raise ValueError("definition_input_owner_missing")
            if typed:
                from .definition_dependencies import assess_owner_evidence, selected_inputs

                typed_manifest = self.input_reader()
                typed_input_snapshot = (
                    typed_manifest.checksum,
                    canonical(typed_manifest.components),
                )
                if isinstance(draft, OwnerBoundResearchDefinitionDraft):
                    selected_inputs(draft, typed_manifest)
                assessment = assess_owner_evidence(
                    draft,
                    typed_manifest,
                    self.binding["input_checksum"],
                    self.fact_reader,
                    confirmed,
                )
            elif isinstance(draft, EvidenceResearchDefinitionDraft):
                from .definition_dependencies import assess_evidence

                assessment = assess_evidence(
                    draft, self.input_reader(), self.binding["input_checksum"], self.fact_reader
                )
            else:
                assessment = assess(draft, self.input_reader(), self.binding["input_checksum"])
        sources = tuple(
            source_read.read(c) for c in dict.fromkeys(e.source_code for e in draft.excerpts)
        )
        refs = tuple(validate(Source, s["material"]).reference for s in sources)
        if any(
            s["code"] != r.code or s["checksum"] != r.checksum
            for s, r in zip(sources, refs, strict=True)
        ):
            raise ValueError("definition_source_material_binding")
        review_allowed = (
            assessment["review_allowed"] if assessment else draft.relation_use == "directional"
        )
        body = dict(
            examination=draft.examination,
            applicability_conditions=draft.applicability,
            interpretation=draft.interpretation if review_allowed else None,
            interpretation_status="installed" if review_allowed else "missing",
            source_refs=refs,
        )
        definition = TaskDefinition(
            reference=dict(
                role="definition",
                code="DEF:" + digest(candidate if located else draft)[7:39],
                version=draft.schema_version.split("/")[1]
                if assessment
                else "2.1.0"
                if located
                else "2.0.0",
                checksum=digest(body),
            ),
            **body,
        )
        source_read.verify()
        if typed:
            final_manifest = self.input_reader()
            if (
                final_manifest.checksum,
                canonical(final_manifest.components),
            ) != typed_input_snapshot:
                raise ValueError("definition_dependency_input_changed")
        if confirmed is not None:
            final_basis = self.confirmed_basis(draft)
            changed = (
                canonical(final_basis) != canonical(confirmed)
                if typed
                else final_basis != confirmed
            )
            if changed:
                raise ValueError("definition_confirmed_requirement_changed")
        return dict(
            schema_version="accepted-research-definition/6.0.0"
            if typed and confirmed is not None
            else "accepted-research-definition/5.0.0"
            if typed
            else "accepted-research-definition/4.0.0"
            if confirmed is not None
            else "accepted-research-definition/3.0.0"
            if isinstance(draft, EvidenceResearchDefinitionDraft)
            else "accepted-research-definition/2.0.0"
            if assessment
            else "accepted-research-definition/1.1.0"
            if located
            else "accepted-research-definition/1.0.0",
            **self.binding,
            draft=candidate if located else draft.model_dump(mode="json"),
            **(
                dict(
                    resolved_draft=draft.model_dump(mode="json"),
                    coordinate_space="source_text_unicode_codepoints",
                )
                if located
                else {}
            ),
            definition=definition.model_dump(mode="json"),
            sources=list(sources),
            qualification="campaign_research_only",
            published=False,
            clinical_acceptance=False,
            medical_entailment="research_judgment_not_deterministically_proven",
            **(dict(confirmed_basis=confirmed) if confirmed is not None else {}),
            **(dict(dependency_assessment=assessment) if assessment else {}),
        )

    def confirmed_basis(self, draft):
        from packages.evaluation_preparation.generic_definition_facts import (
            ConfirmedGenericMeasurements,
        )
        from packages.multimodal_research.component_observations import ComponentObservations
        from packages.multimodal_research.confirmed_numeric_observations import (
            ConfirmedNumericObservations,
        )
        from packages.multimodal_research.definition_context import public_examination

        owner = self.confirmed_observations
        if type(owner) not in {
            ComponentObservations,
            ConfirmedNumericObservations,
            ConfirmedGenericMeasurements,
        }:
            raise ValueError("definition_confirmed_requirement_owner_missing")
        index = int(draft.requirement_alias.split(":")[1])
        if index >= len(self.requirement_ids):
            raise ValueError("definition_confirmed_requirement_alias")
        generic = type(owner) is ConfirmedGenericMeasurements
        if not generic and (
            owner.base.manifest.checksum != self.binding["input_checksum"]
            or owner.base.task.target_code != self.binding["target"]
        ):
            raise ValueError("definition_confirmed_requirement_scope")
        basis = (
            owner.dependency_basis(self.requirement_ids[index])
            if isinstance(draft, OwnerBoundConfirmedInterpretationDraft)
            else owner.interpretation_basis(self.requirement_ids[index])
        )
        if generic and {k: basis["origin"][k] for k in ("input_checksum", "target")} != {
            k: self.binding[k] for k in ("input_checksum", "target")
        }:
            raise ValueError("definition_confirmed_requirement_scope")
        row = basis["requirement"]
        if (
            draft.examination.strip().casefold() != public_examination(row).strip().casefold()
            or tuple(draft.modalities) != tuple(row["input_modalities"])
            or (draft.method, draft.unit) != (basis["method"], basis["unit"])
        ):
            raise ValueError("definition_confirmed_requirement_method")
        return basis

    def project_confirmed(self, receipt):
        """Same requirement, new interpretation; the original fact is not replaced."""
        if receipt.get("schema_version") not in CONFIRMED_RECEIPTS:
            raise ValueError("definition_confirmed_receipt_version")
        if canonical(self._build(receipt["draft"])) != canonical(receipt):
            raise ValueError("definition_confirmed_receipt_binding")
        return self._confirmed_projection(receipt)

    def read_confirmed_projection(self, key):
        """One current read plus pure projection, not dispatch authority or a cache.

        Reconstruction still brackets source/input reads with original confirmed
        fact checks. Only the immediate duplicate reconstruction is removed.
        Caller supplied receipts must continue through project_confirmed().
        """
        receipt = self.read(key, historical=True)
        if receipt.get("schema_version") not in CONFIRMED_RECEIPTS:
            raise ValueError("definition_confirmed_receipt_version")
        if canonical(self._build(receipt["draft"])) != canonical(receipt):
            raise ValueError("definition_confirmed_receipt_binding")
        return receipt, self._confirmed_projection(receipt)

    @staticmethod
    def _confirmed_projection(receipt):
        basis = receipt["confirmed_basis"]
        draft = receipt.get("resolved_draft", receipt["draft"])
        assessment = receipt["dependency_assessment"]
        return dict(
            basis["requirement"],
            definition=validate(TaskDefinition, receipt["definition"]),
            basis_kind="quantitative" if draft["relation"] == "quantitative" else "qualitative",
            definition_receipt_checksum=digest(receipt),
            dependency_assessment=assessment,
            acquisition_allowed=assessment["acquisition_allowed"],
            review_allowed=assessment["review_allowed"],
            expected_method=basis["method"],
            expected_unit=basis["unit"],
        )

    def accept(self, key, draft):
        value = self.build(draft)
        # All source I/O happens before the short immutable Runtime transaction.
        with self.documents.session():
            self.guard()
            self.documents.write("definition:" + key, (value,))
        # A returned current qualification requires a fresh owner read after commit.
        return self.read(key)

    def read(self, key, *, historical=False):
        saved = self.documents.read("definition:" + key)
        if saved is None:
            raise ValueError("definition_not_saved")
        value = saved[0]
        if {k: value[k] for k in self.binding} != self.binding:
            raise ValueError("definition_campaign_scope")
        if not historical:
            rebuilt = self.build(value["draft"])
            changed = (
                canonical(rebuilt) != canonical(value)
                if value.get("schema_version") in OWNER_BOUND_RECEIPTS
                else rebuilt != value
            )
            if changed:
                raise ValueError("definition_current_binding")
        return value

    def read_for_report(self, key):
        """Read a saved receipt with current source checks, even after run stop.

        This does not replace read()/build()/accept() for current consumption.
        The original guard still controls all acceptance and execution paths.
        """
        value = self.read(key, historical=True)
        rebuilt = self._build(value["draft"])
        if (
            canonical(rebuilt) != canonical(value)
            if value.get("schema_version") in OWNER_BOUND_RECEIPTS
            else rebuilt != value
        ):
            raise ValueError("definition_current_binding")
        return value


def requirement_rows(accepted, manifest, *, fact_reader=None):
    """Owner-derived current input projection, not a list of fixed disease slots."""
    if accepted.get("schema_version") in CONFIRMED_RECEIPTS:
        # Do not turn a fact-interpretation receipt into an independently added
        # acquisition row. A same-ID revision needs the original projection port.
        raise ValueError("definition_confirmed_projection_owner_required")
    typed = accepted.get("schema_version") == "accepted-research-definition/5.0.0"
    evidence = typed or accepted.get("schema_version") == "accepted-research-definition/3.0.0"
    scoped = evidence or accepted.get("schema_version") == "accepted-research-definition/2.0.0"
    draft = validate(
        OwnerBoundResearchDefinitionDraft
        if typed
        else EvidenceResearchDefinitionDraft
        if evidence
        else ScopedResearchDefinitionDraft
        if scoped
        else ResearchDefinitionDraft,
        accepted.get("resolved_draft", accepted["draft"]),
    )
    assessment = None
    if scoped:
        from .definition_dependencies import assess

        if typed:
            from .definition_dependencies import assess_owner_evidence

            assessment = assess_owner_evidence(
                draft, manifest, accepted["input_checksum"], fact_reader, None
            )
        elif evidence:
            from .definition_dependencies import assess_evidence

            assessment = assess_evidence(draft, manifest, accepted["input_checksum"], fact_reader)
        else:
            assessment = assess(draft, manifest, accepted["input_checksum"])
        if (
            canonical(assessment) != canonical(accepted["dependency_assessment"])
            if typed
            else assessment != accepted["dependency_assessment"]
        ) or (accepted["definition"]["interpretation_status"] == "installed") != assessment[
            "review_allowed"
        ]:
            raise ValueError("definition_dependency_projection_binding")
    if draft.dependencies and not scoped:
        raise ValueError("definition_dependency_unresolved")
    if len(draft.modalities) != 1:
        raise ValueError("definition_joint_input_method_not_installed")
    semantic = digest(
        (
            draft.examination.strip().casefold(),
            draft.modalities,
            draft.input_concepts,
            draft.acquisition,
            draft.method,
            draft.unit,
        )
    )
    components = [
        c
        for c in manifest.components
        if c.modality == draft.modalities[0]
        and c.status != "withheld"
        and (draft.acquisition == "vision" or c.concept in draft.input_concepts)
    ]
    if typed:
        from .definition_dependencies import selected_inputs

        components = selected_inputs(draft, manifest)
    rows = []
    for c in components or [None]:
        present = c is not None and c.status in {"obtained", "partial"}
        if (
            present
            and draft.acquisition == "vision"
            and c.kind not in {"image", "volume", "image_series"}
        ):
            raise ValueError("definition_input_method_mismatch")
        identifier = "discovered:" + semantic[7:27] + (":" + c.id if c else "")
        rows.append(
            dict(
                id=identifier,
                component=c.id if c else None,
                category="definition:" + semantic[7:27],
                role="association" if draft.relation == "context" else "diagnostic",
                definition=validate(TaskDefinition, accepted["definition"]),
                input_modalities=draft.modalities,
                basis_kind="quantitative" if draft.relation == "quantitative" else "qualitative",
                missing_reason="definition_acquisition_dependency_unresolved"
                if assessment and not assessment["acquisition_allowed"] and present
                else "not_yet_observed"
                if present
                else "patient_input_missing:" + draft.modalities[0],
                acquisition_method=draft.acquisition,
                evidence_unit=semantic,
                parent="DISCOVERED",
                definition_receipt_checksum=digest(accepted),
                semantic_identity=semantic,
                expected_method=draft.method,
                expected_unit=draft.unit,
                **(
                    dict(
                        dependency_assessment=assessment,
                        acquisition_allowed=assessment["acquisition_allowed"],
                        review_allowed=assessment["review_allowed"],
                    )
                    if assessment
                    else {}
                ),
            )
        )
    return rows
