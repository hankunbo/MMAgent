"""Original Knowledge search/archive/index ports for bounded research definitions.

Search hits alone never become definitions. New network fetches use an installed
controlled connector; this class itself neither reads credentials nor opens HTTP.
"""

from typing import Literal

from pydantic import Field, model_validator

from packages.grounding.models import Value, digest, validate
from packages.knowledge_index.models import Filters, IndexConfig


def checked_fetch_decisions(plan, hits):
    """Validate a saved original-owner decision before deferring any wire work."""
    decisions = plan.get("decisions")
    indexes = plan.get("index_keys")
    if (
        not isinstance(indexes, list)
        or any(type(k) is not str or not k for k in indexes)
        or len(indexes) != len(set(indexes))
        or not isinstance(decisions, list)
        or len(decisions) != len(hits)
    ):
        raise ValueError("definition_fetch_order_binding")
    for position, decision in enumerate(decisions):
        if (
            not isinstance(decision, dict)
            or type(decision.get("position")) is not int
            or decision["position"] != position
            or decision.get("hit_ref") != hits[position]["reference"]
            or type(decision.get("already_acquired_identity")) is not bool
            or type(decision.get("ambiguous_source_identity")) is not bool
            or not isinstance(decision.get("matched"), list)
            or not isinstance(decision.get("conflicts"), list)
        ):
            raise ValueError("definition_fetch_order_binding")
        matches = decision["matched"]
        for match in matches:
            if (
                not isinstance(match, dict)
                or not isinstance(match.get("index_key"), str)
                or match["index_key"] not in indexes
                or not isinstance(match.get("source_ref"), dict)
                or not isinstance(match["source_ref"].get("code"), str)
                or not isinstance(match.get("source_version"), dict)
                or not isinstance(match["source_version"].get("code"), str)
            ):
                raise ValueError("definition_fetch_order_binding")
        ambiguous = len({m["source_ref"]["code"] for m in matches}) > 1
        known = bool(matches) and not decision["conflicts"] and not ambiguous
        if decision["ambiguous_source_identity"] != ambiguous or (
            decision["already_acquired_identity"] != known
        ):
            raise ValueError("definition_fetch_order_binding")
    if plan["order"] != sorted(
        range(len(hits)), key=lambda i: decisions[i]["already_acquired_identity"]
    ):
        raise ValueError("definition_fetch_order_binding")
    return decisions


class PassagePlan(Value):
    """Public, local-only recall plan; each alternative still uses original FTS."""

    schema_version: Literal["definition-passage-plan/1.0.0"] = "definition-passage-plan/1.0.0"
    queries: tuple[str, ...] = Field(min_length=1, max_length=6)
    max_per_document: int = Field(default=3, ge=1, le=3)
    context_radius: int = Field(default=1, ge=0, le=1)
    max_characters: int = Field(default=12000, ge=1000, le=16000)

    @model_validator(mode="after")
    def public_queries(self):
        from packages.knowledge_search.privacy import safe_text

        if len(set(self.queries)) != len(self.queries) or any(
            len(q) > 120 or not safe_text(q) for q in self.queries
        ):
            raise ValueError("definition_passage_query_scope")
        return self


class IntentPassagePlan(PassagePlan):
    """Bounded retrieval coverage, not a claim that a passage proves an intent."""

    schema_version: Literal["definition-passage-plan/2.0.0"] = "definition-passage-plan/2.0.0"
    query_intents: dict[str, Literal["examination", "target", "alternative"]]

    @model_validator(mode="after")
    def exact_query_intents(self):
        if set(self.query_intents) != set(self.queries):
            raise ValueError("definition_passage_intent_binding")
        return self


class DiscoursePassagePlan(IntentPassagePlan):
    """Structural recall eligibility only; never a medical relevance decision."""

    schema_version: Literal["definition-passage-plan/3.0.0"] = "definition-passage-plan/3.0.0"


class BalancedPassagePlan(DiscoursePassagePlan):
    """Source diversity after intent coverage; neither novelty nor medical quality."""

    schema_version: Literal["definition-passage-plan/4.0.0"] = "definition-passage-plan/4.0.0"


class ExplanatoryPassagePlan(BalancedPassagePlan):
    """Use body prose/list/table anchors; headings/captions remain context leads.

    Structural eligibility is not medical entailment. Original current reads and
    the definition owner still decide whether any selected passage is usable.
    """

    schema_version: Literal["definition-passage-plan/5.0.0"] = "definition-passage-plan/5.0.0"


class BoundedMaterialPassagePlan(ExplanatoryPassagePlan):
    """A wider current-index pool with the unchanged total character ceiling.

    This version is local-only. It changes available context capacity, not the
    retrieval ranking, source qualification, or medical interpretation.
    """

    schema_version: Literal["definition-passage-plan/6.0.0"] = "definition-passage-plan/6.0.0"
    max_per_document: int = Field(default=8, ge=1, le=8)


class WholeBlockMaterialPassagePlan(BoundedMaterialPassagePlan):
    """Complete research blocks through the original owner, at most4000 characters."""

    schema_version: Literal["definition-passage-plan/7.0.0"] = "definition-passage-plan/7.0.0"


def read_passage_plan(value):
    if isinstance(value, PassagePlan):
        value = value.model_dump(mode="json")
    cls = (
        WholeBlockMaterialPassagePlan
        if value.get("schema_version") == "definition-passage-plan/7.0.0"
        else BoundedMaterialPassagePlan
        if value.get("schema_version") == "definition-passage-plan/6.0.0"
        else ExplanatoryPassagePlan
        if value.get("schema_version") == "definition-passage-plan/5.0.0"
        else BalancedPassagePlan
        if value.get("schema_version") == "definition-passage-plan/4.0.0"
        else DiscoursePassagePlan
        if value.get("schema_version") == "definition-passage-plan/3.0.0"
        else IntentPassagePlan
        if value.get("schema_version") == "definition-passage-plan/2.0.0"
        else PassagePlan
    )
    return validate(cls, value)


class DefinitionMaterialUnavailable(ValueError):
    """A known source-material gap, not a transport or identity failure."""


class DefinitionAdmissionStopped(RuntimeError):
    """Keep owner admission errors outside the HTTP parser's ValueError category."""

    def __init__(self, original):
        self.original = original
        super().__init__(original.code)


def save_fetch(request, store):
    """Journal public bytes in the original immutable object store, not Runtime JSON.

    TransientResponse intentionally excludes body from model_dump; recording only
    that dump loses the material on recovery. A crash may leave an unreferenced
    immutable object, but never authorizes a second remote fetch.
    """
    from packages.knowledge_archive import FetchInput

    request = FetchInput.model_validate(request)
    value = request.model_dump(mode="json")

    def persist(response, target):
        staged = store.stage(response.body)
        target["body_object"] = store.promote(staged, response.response_checksum)

    persist(request.document.response, value["document"]["response"])
    for response, target in zip(
        request.discovery_responses, value["discovery_responses"], strict=True
    ):
        persist(response, target)
    return dict(schema_version="research-definition-fetch/1.0.0", payload=value)


def restore_fetch(saved, store):
    """Revalidate original types, byte counts and checksums without a remote call."""
    import copy
    import json
    from datetime import datetime

    from packages.knowledge_archive import Bibliography, FetchInput
    from packages.knowledge_connectors import FetchedDocument, SourceLocator
    from packages.knowledge_connectors.ports import RequestMetadata, TransientResponse

    if saved.get("schema_version") != "research-definition-fetch/1.0.0":
        raise ValueError("definition_fetch_version")
    value = copy.deepcopy(saved["payload"])

    def response(raw):
        uri = raw.pop("body_object")
        raw["body"] = store.read_bounded(uri, 4 * 1024 * 1024)
        raw["metadata"] = RequestMetadata.model_validate_json(json.dumps(raw["metadata"]))
        return TransientResponse.model_validate(raw)

    doc = value["document"]
    doc["response"] = response(doc["response"])
    doc["locator"] = SourceLocator.model_validate_json(json.dumps(doc["locator"]))
    value["document"] = FetchedDocument.model_validate(doc)
    value["bibliography"] = Bibliography.model_validate_json(json.dumps(value["bibliography"]))
    value["discovery_responses"] = tuple(response(r) for r in value["discovery_responses"])
    if value["last_modified"] is not None:
        value["last_modified"] = datetime.fromisoformat(value["last_modified"])
    return FetchInput.model_validate(value)


class KnowledgeDefinitionSources:
    def __init__(
        self,
        *,
        repository,
        store,
        index_keys=(),
        fetch_hit=None,
        search_public=None,
        applicability_refs=(),
        filters=None,
        failure_policy=None,
        material_policy=None,
        passage_plans=None,
    ):
        if failure_policy not in {None, "technical_backlog_no_retry"}:
            raise ValueError("definition_source_failure_policy")
        self.failure_policy = failure_policy
        if material_policy not in {
            None,
            "xml-scripts-whole-text/1.0.0",
            "xml-jats-whole-text/1.0.0",
            "xml-jats-citations-whole-text/1.0.0",
            "xml-jats-citations-windowed/1.0.0",
        }:
            raise ValueError("definition_material_policy")
        self.material_policy = material_policy
        self.passage_plans = {q: read_passage_plan(p) for q, p in (passage_plans or {}).items()}
        if self.passage_plans and len({type(p) for p in self.passage_plans.values()}) != 1:
            raise ValueError("definition_mixed_passage_plan_versions")
        self.intent_coverage = bool(self.passage_plans) and all(
            isinstance(p, IntentPassagePlan) for p in self.passage_plans.values()
        )
        self.index_receipts = {}
        self.before_read = lambda: None
        self.repository, self.store = repository, store
        self.index_keys = tuple(index_keys)
        self.fetch_hit, self.search_public = fetch_hit, search_public
        self.applicability_refs = tuple(applicability_refs)
        self.hits = {}
        self.query_backlog = []
        # This population tag scopes public text processing, never a patient
        # population or clinical applicability assertion. Original SQL still gates it.
        self.filters = filters or Filters(population=("research_definition_material_only",))

    def index_scope(self, archived):
        if self.applicability_refs:
            return self.applicability_refs
        from db.knowledge import models as m
        from packages.grounding.models import digest, validate

        expected = Filters(population=("research_definition_material_only",))
        if self.filters != expected:
            raise ValueError("definition_index_scope_not_installed")
        reference = validate(m.KnowledgeReference, archived["version_ref"])
        snapshot = self.repository.get(reference)
        if snapshot.withdrawn or snapshot.status == m.KnowledgeStatus.DEPRECATED:
            raise ValueError("definition_source_unavailable")
        condition = m.seal(
            m.ApplicabilityCondition,
            code="EVIDENCE:DEFINITION_MATERIAL:" + digest(reference)[7:39].upper(),
            version="1.0.0",
            synthetic=snapshot.record.synthetic,
            source_refs=(reference,),
            population=expected.population,
            scope_note=(
                "Public text processing for candidate research definitions only. "
                "No assertion of patient applicability, clinical validity or publication."
            ),
        )
        self.repository.put(condition)
        return (condition.reference(),)

    def restore(self, materials):
        for s in materials:
            self.hits[s["code"]] = s["hit"]

    def restore_indexes(self, receipts):
        """Restore confirmed addresses, never current retrieval permission.

        The caller supplies receipts from its original immutable query operation.
        Historical owner replay verifies their bytes; each later FTS/material read
        still invokes the current Knowledge owner, including withdrawal checks.
        """
        if not self.intent_coverage:
            raise ValueError("definition_index_pool_version")
        windowed = self.material_policy == "xml-jats-citations-windowed/1.0.0"
        if not isinstance(receipts, (list, tuple)) or len(receipts) > (8 if windowed else 2):
            raise ValueError("definition_index_pool_bound")
        checked = {}
        for expected in receipts:
            self.before_read()
            actual = self.repository.replay_document_index(expected["key"], self.store)
            if digest(actual) != digest(expected) or actual.get("status") != "indexed":
                raise ValueError("definition_index_receipt_binding")
            checked[actual["key"]] = actual
        if windowed:
            by_source = {}
            for actual in checked.values():
                source = digest(actual["source_version_ref"])
                by_source[source] = by_source.get(source, 0) + 1
            if len(by_source) > 2 or any(n > 4 for n in by_source.values()):
                raise ValueError("definition_index_pool_bound")
        keys = tuple(dict.fromkeys((*self.index_keys, *checked)))
        if len(keys) > 10:
            raise ValueError("definition_index_pool_bound")
        self.index_keys = keys
        self.index_receipts.update(checked)

    def _material_options(self):
        if self.passage_plans and all(
            isinstance(p, WholeBlockMaterialPassagePlan) for p in self.passage_plans.values()
        ):
            return {"material_profile": "research-definition-material/1.1.0"}
        return {}

    def read(self, code):
        if code not in self.hits:
            raise ValueError("definition_source_not_acquired")
        return self.repository.read_definition_material(
            self.hits[code], self.store, filters=self.filters, **self._material_options()
        )

    def read_many(self, codes):
        """Exact ordered current reads, at most sixteen per original owner call."""
        if not isinstance(codes, (list, tuple)) or len(codes) > 64:
            raise ValueError("definition_material_batch_bound")
        if any(code not in self.hits for code in codes):
            raise ValueError("definition_source_not_acquired")
        result = []
        for start in range(0, len(codes), 16):
            result.extend(
                self.repository.read_definition_materials(
                    [self.hits[code] for code in codes[start : start + 16]],
                    self.store,
                    filters=self.filters,
                    **self._material_options(),
                )
            )
        return result

    def _query(self, keys, query, limit):
        from packages.knowledge_claims.evidence import Rejected

        result = []
        for key in keys:
            value = self.repository.query_candidate_index(key, query, self.filters)
            for hit in value["hits"]:
                self.hits[hit["chunk_ref"]["code"]] = hit
                try:
                    source = self.read(hit["chunk_ref"]["code"])
                except Rejected as exc:
                    if str(exc) not in {"incomplete_context", "table_unavailable"}:
                        raise
                    # A search slice can be valid retrieval but not complete
                    # medical context. Keep the rejection; try other ranked hits.
                    self.hits.pop(hit["chunk_ref"]["code"], None)
                    self.query_backlog.append(dict(reason="source_chunk_" + str(exc), hit=hit))
                    continue
                if source["code"] not in {s["code"] for s in result}:
                    result.append(source)
                if len(result) >= limit:
                    return result
        return result

    def _context_query(self, keys, query, limit, *, plans=None):
        """Bounded union of aliases plus exact same-section neighboring chunks.

        Multiple matches never create additional sources/independent votes. No
        concatenated or rewritten medical text; every paragraph keeps its owner ID.
        """
        plan = self.passage_plans[query]
        found, counts, characters, trace = {}, {}, 0, []
        balanced = isinstance(plan, BalancedPassagePlan)
        source_keys = {}
        if balanced:
            self.retrieval["candidate_decisions"] = []

        def decision(code, reason, *, current_material_read):
            if balanced:
                self.retrieval["candidate_decisions"].append(
                    dict(
                        code=code,
                        source_identity=source_keys[code],
                        reason=reason,
                        current_material_read=current_material_read,
                        medical_entailment_proven=False,
                    )
                )

        anchors = []
        intentions = {k: [] for k in ("examination", "target", "alternative")}
        provenance = {}
        alternatives = (
            list(dict.fromkeys((q, p.query_intents[q]) for p in plans for q in p.queries))
            if plans is not None
            else [
                (q, plan.query_intents[q] if isinstance(plan, IntentPassagePlan) else None)
                for q in plan.queries
            ]
        )
        batches = {}
        batch_reader = getattr(self.repository, "query_candidate_aliases", None)
        if callable(batch_reader):
            queries = tuple(dict.fromkeys(q for q, _ in alternatives))
            for key in keys:
                batches[key] = {}
                for start in range(0, len(queries), 32):
                    chunk = queries[start : start + 32]
                    batches[key].update(
                        zip(
                            chunk,
                            batch_reader(key, chunk, self.filters, before_read=self.before_read),
                            strict=True,
                        )
                    )
            self.retrieval["local_query_execution"] = dict(
                schema_version="current-candidate-alias-batch/1.0.0",
                indexes=len(keys),
                queries_per_index=len(queries),
                current_reads_per_index=(len(queries) + 31) // 32,
                consumption_revalidation_required=True,
            )
        for alternative, intent in alternatives:
            for key in keys:
                self.before_read()
                result = (
                    batches[key][alternative]
                    if batches
                    else self.repository.query_candidate_index(
                        key, alternative, self.filters, fts_single_pass=True
                    )
                )
                trace.append(dict(index_key=key, query=alternative, hits=len(result["hits"])))
                for rank, hit in enumerate(result["hits"], start=1):
                    if balanced:
                        source_ref = hit.get("source_version_ref")
                        if not isinstance(source_ref, dict) or not source_ref:
                            raise ValueError("definition_pool_source_identity")
                        code = hit["chunk_ref"]["code"]
                        identity = digest(source_ref)
                        if source_keys.setdefault(code, identity) != identity:
                            raise ValueError("definition_pool_source_identity")
                    if isinstance(plan, IntentPassagePlan):
                        code = hit["chunk_ref"]["code"]
                        provenance.setdefault(code, []).append(
                            dict(intent=intent, query=alternative, index_key=key, rank=rank)
                        )
                        if code not in {h["chunk_ref"]["code"] for h in intentions[intent]}:
                            intentions[intent].append(hit)
                    if hit["chunk_ref"]["code"] not in {h["chunk_ref"]["code"] for h in anchors}:
                        anchors.append(hit)

        if isinstance(plan, IntentPassagePlan):
            # Rank is only ordinal within each query, never a comparable medical
            # strength. Round-robin coverage precedes document quotas/context.
            anchors, seen = [], set()
            for index in range(max(map(len, intentions.values()), default=0)):
                for candidates in intentions.values():
                    if index < len(candidates):
                        hit = candidates[index]
                        code = hit["chunk_ref"]["code"]
                        if code not in seen:
                            seen.add(code)
                            anchors.append(hit)

        from packages.knowledge_claims.evidence import Rejected

        def add(material, origin):
            nonlocal characters
            code, size = material["code"], len(material["text"])
            if balanced:
                identity = digest(material["source_version"])
                if source_keys.setdefault(code, identity) != identity:
                    raise ValueError("definition_pool_source_identity")
            # A bibliography entry is a lead to another document, not that
            # document's obtained explanatory text. Keep it in the backlog.
            sections = material.get("position", {}).get("section_path", ())
            if (
                isinstance(plan, ExplanatoryPassagePlan)
                and origin == "retrieval_match"
                and material.get("position", {}).get("kind") not in {"paragraph", "list", "table"}
            ):
                self.query_backlog.append(dict(reason="non_explanatory_anchor", code=code))
                decision(code, "non_explanatory_anchor", current_material_read=True)
                return False
            if any(
                s.strip().casefold() in {"references", "bibliography", "参考文献"} for s in sections
            ):
                self.query_backlog.append(
                    dict(reason="bibliography_not_definition_text", code=code)
                )
                decision(code, "bibliography_not_definition_text", current_material_read=True)
                return False
            if isinstance(plan, DiscoursePassagePlan) and any(
                s.strip().casefold()
                in {"abbreviations", "list of abbreviations", "缩略词", "缩写词表"}
                for s in sections
            ):
                # The original text remains indexed/readable. Only this explicit
                # new plan excludes a glossary from limited explanatory slots.
                # Do not inspect medical words, infer direction, or rewrite text.
                self.query_backlog.append(
                    dict(reason="non_discourse_section_not_definition_text", code=code)
                )
                decision(
                    code, "non_discourse_section_not_definition_text", current_material_read=True
                )
                return False
            doc = digest(material["source_version"])
            if code in found:
                decision(code, "duplicate_material", current_material_read=True)
                return False
            if len(found) >= limit or counts.get(doc, 0) >= plan.max_per_document:
                decision(code, "material_or_document_quota", current_material_read=True)
                return False
            if characters + size > plan.max_characters:
                self.query_backlog.append(dict(reason="source_context_character_budget", code=code))
                decision(code, "source_context_character_budget", current_material_read=True)
                return False
            found[code] = material
            self.hits[code] = material["hit"]
            counts[doc] = counts.get(doc, 0) + 1
            characters += size
            selection = dict(code=code, origin=origin)
            if isinstance(plan, IntentPassagePlan):
                selection["query_matches"] = provenance.get(code, [])
                selection["intent_is_entailment"] = False
            self.retrieval["selections"].append(selection)
            decision(code, "selected", current_material_read=True)
            return True

        context_anchors, covered_intents = [], set()
        while anchors:
            if len(found) >= limit:
                break
            if isinstance(plan, IntentPassagePlan):

                def coverage_priority(hit):
                    matches = provenance[hit["chunk_ref"]["code"]]
                    return min(
                        (
                            list(intentions).index(m["intent"])
                            for m in matches
                            if m["intent"] not in covered_intents
                        ),
                        default=3,
                    )

                # A rejected bibliography/incomplete hit must not consume the
                # intention's turn. Try its next ranked candidate first.
                def priority(hit):
                    coverage = coverage_priority(hit)
                    if not balanced:
                        return (coverage,)
                    code = hit["chunk_ref"]["code"]
                    # Per-query rank is only a retrieval tie-breaker, never a
                    # comparable medical strength. Stable source/code ties do
                    # not depend on the order indexes were installed.
                    return (
                        coverage,
                        counts.get(source_keys[code], 0),
                        min(m["rank"] for m in provenance[code]),
                        source_keys[code],
                        code,
                    )

                index = min(range(len(anchors)), key=lambda i: priority(anchors[i]))
                hit = anchors.pop(index)
            else:
                hit = anchors.pop(0)
            if isinstance(plan, ExplanatoryPassagePlan) and hit.get("position", {}).get(
                "kind"
            ) not in {"paragraph", "list", "table"}:
                # Filtering cannot grant source authorization. Do not spend an
                # expensive current-content reconstruction on an unused title.
                code = hit["chunk_ref"]["code"]
                self.query_backlog.append(dict(reason="non_explanatory_anchor", code=code))
                decision(code, "non_explanatory_anchor", current_material_read=False)
                continue
            # A full document quota cannot produce another selected paragraph.
            # Skip before expensive owner reconstruction; no cached source is
            # substituted for a current read of material that will be consumed.
            ref = hit.get("source_version_ref")
            if ref and counts.get(digest(ref), 0) >= plan.max_per_document:
                decision(
                    hit["chunk_ref"]["code"],
                    "document_quota_not_evaluated",
                    current_material_read=False,
                )
                continue
            self.before_read()
            try:
                material = self.repository.read_definition_material(
                    hit, self.store, filters=self.filters, **self._material_options()
                )
            except Rejected as exc:
                material_gaps = {"incomplete_context", "table_unavailable"}
                if isinstance(plan, BoundedMaterialPassagePlan):
                    # The owner cannot supply an oversized paragraph as a
                    # claim. Do not truncate it or spend a Provider request;
                    # retain this explicit material gap and inspect the next.
                    material_gaps.add("claim_text_limit")
                if isinstance(plan, WholeBlockMaterialPassagePlan):
                    material_gaps.add("definition_source_requires_bounded_chunk")
                if str(exc) not in material_gaps:
                    raise
                reason = "source_chunk_" + str(exc)
                self.query_backlog.append(dict(reason=reason, hit=hit))
                decision(
                    hit["chunk_ref"]["code"],
                    reason,
                    current_material_read=True,
                )
                continue
            if not add(material, "retrieval_match"):
                continue
            if isinstance(plan, IntentPassagePlan):
                covered_intents.update(m["intent"] for m in provenance[material["code"]])
                context_anchors.append((hit, material))
                continue
            if (
                plan.context_radius
                and len(found) < limit
                and counts.get(digest(material["source_version"]), 0) < plan.max_per_document
            ):
                self.before_read()
                neighbors = self.repository.read_definition_context(
                    hit,
                    self.store,
                    filters=self.filters,
                    radius=plan.context_radius,
                    **self._material_options(),
                )
                for neighbor in neighbors:
                    add(neighbor, "same_section_context_not_independent_entailment")
        for hit in anchors:
            decision(
                hit["chunk_ref"]["code"],
                "material_limit_not_evaluated",
                current_material_read=False,
            )
        if isinstance(plan, IntentPassagePlan) and plan.context_radius:
            for hit, material in context_anchors:
                if len(found) >= limit:
                    break
                if counts.get(digest(material["source_version"]), 0) >= plan.max_per_document:
                    continue
                self.before_read()
                for neighbor in self.repository.read_definition_context(
                    hit,
                    self.store,
                    filters=self.filters,
                    radius=plan.context_radius,
                    **self._material_options(),
                ):
                    add(neighbor, "same_section_context_not_independent_entailment")
        self.retrieval["attempts"].extend(trace)
        return list(found.values())

    def checked_material_limit(self, limit):
        maximum = (
            16
            if self.passage_plans
            and all(isinstance(p, BoundedMaterialPassagePlan) for p in self.passage_plans.values())
            else 5
        )
        if type(limit) is not int or not 1 <= limit <= maximum:
            raise ValueError("definition_material_limit")

    def reconcile_pool(self, *, queries, limit):
        """Local-only selection over the complete confirmed index pool.

        No search/fetch fallback is permitted here. The caller must journal this
        revision before supplying its materials to the definition provider.
        Shared quotas apply once across all plans, not once per public query.
        """
        self.checked_material_limit(limit)
        if (
            not isinstance(queries, (tuple, list))
            or not 1 <= len(queries) <= 2
            or len(set(queries)) != len(queries)
        ):
            raise ValueError("definition_pool_query_bound")
        plans = [self.passage_plans[q] for q in queries]
        if any(not isinstance(p, DiscoursePassagePlan) for p in plans):
            raise ValueError("definition_pool_plan_version")
        if (
            any(isinstance(p, BalancedPassagePlan) for p in plans)
            and len({type(p) for p in plans}) != 1
        ):
            raise ValueError("definition_pool_plan_version")
        quotas = {(p.max_per_document, p.context_radius, p.max_characters) for p in plans}
        if len(quotas) != 1:
            raise ValueError("definition_pool_quota_binding")
        if len(self.index_keys) > 10:
            raise ValueError("definition_index_pool_bound")
        self.query_backlog = []
        self.retrieval = dict(
            queries=list(queries),
            policy_checksum=digest(plans),
            index_keys=list(self.index_keys),
            attempts=[],
            selections=[],
            medical_entailment_proven=False,
        )
        if self._material_options():
            self.retrieval["material_reader"] = dict(
                self._material_options(), max_block_characters=4000, complete_blocks_only=True
            )
        materials = self._context_query(self.index_keys, queries[0], limit, plans=plans)
        self.before_read()
        return dict(
            schema_version="research-definition-pool/1.0.0",
            materials=materials,
            backlog=list(self.query_backlog),
            retrieval=dict(self.retrieval, selected_codes=[m["code"] for m in materials]),
        )

    def search(self, *, query, key, limit, operation, retrieval_mode=None):
        """Existing current material first. operation journals *each* external action.

        Installed connector implementations must disable internal retries or use
        operation for every attempt. Missing network wiring is explicit backlog,
        never silently promoted title/abstract similarity.
        """
        if retrieval_mode not in {
            None,
            "local-and-public-for-owner-gap/1.0.0",
            "unseen-first-for-owner-gap/1.0.0",
        }:
            raise ValueError("definition_retrieval_mode")
        self.checked_material_limit(limit)
        if self.passage_plans and query not in self.passage_plans:
            raise ValueError("definition_passage_plan_missing")
        local_only = isinstance(self.passage_plans.get(query), BoundedMaterialPassagePlan)
        if local_only and retrieval_mode is not None:
            raise ValueError("definition_bounded_pool_local_only")
        self.query_backlog = []
        acquired_indexes = []
        self.retrieval = dict(
            query=query,
            policy_checksum=digest(self.passage_plans[query]) if self.passage_plans else None,
            attempts=[],
            selections=[],
            medical_entailment_proven=False,
        )
        local_query = self._context_query if self.passage_plans else self._query
        local_materials = []
        fetch_order = None

        def response(materials, reused, backlog):
            if retrieval_mode:
                # Reserve selection capacity for newly obtained material. Keep
                # local context as fallback, never treat recall as entailment.
                selected = {s["code"]: s for s in materials}
                for material in local_materials:
                    if len(selected) >= limit:
                        break
                    selected.setdefault(material["code"], material)
                materials = list(selected.values())
            result = dict(materials=materials, reused=reused, backlog=backlog)
            if retrieval_mode:
                result["discovery"] = dict(
                    mode=retrieval_mode,
                    local_materials=len(local_materials),
                    local_codes=[s["code"] for s in local_materials],
                    medical_entailment_proven=False,
                )
                if fetch_order is not None:
                    result["discovery"]["fetch_order"] = fetch_order
            if self.intent_coverage:
                result.update(
                    schema_version="research-definition-search/1.1.0",
                    acquired_indexes=acquired_indexes,
                )
            if self.passage_plans:
                result["retrieval"] = dict(
                    self.retrieval, selected_codes=[s["code"] for s in materials]
                )
            return result

        found = local_query(self.index_keys, query, limit)
        local_materials = found
        if local_only:
            return response(found, True, list(self.query_backlog))
        if found and not retrieval_mode:
            return response(found, True, list(self.query_backlog))
        if self.search_public is None or self.fetch_hit is None:
            return response(
                [], False, [*self.query_backlog, dict(reason="public_connector_not_installed")]
            )
        hits = operation(
            key + ":search",
            "search_group",
            lambda: self.search_public(query=query, key=key, limit=limit),
        )
        if isinstance(hits, dict) and hits.get("status") == "public_budget_exhausted":
            return response([], False, [*self.query_backlog, hits])
        if isinstance(hits, dict) and hits.get("status") == "technical_backlog":
            return response([], False, [hits])
        found, backlog = [], []
        order = list(range(len(hits[:limit])))
        decisions = None
        if retrieval_mode == "unseen-first-for-owner-gap/1.0.0":

            def rank():
                self.before_read()
                value = self.repository.rank_definition_discovery_hits(
                    hits=hits[:limit], index_keys=self.index_keys, object_store=self.store
                )
                if value.get("index_keys") != list(dict.fromkeys(self.index_keys)):
                    raise ValueError("definition_fetch_order_binding")
                return value

            # Freeze ordering before any fetch: resumed indexes may grow, but an
            # original fetch key must never be rebound to a differently ranked hit.
            fetch_order = operation(key + ":fetch-order", "fetch_order", rank)
            if (
                fetch_order.get("schema_version") != "definition-fetch-order/1.0.0"
                or fetch_order.get("hits") != hits[:limit]
                or fetch_order.get("eligibility_granted") is not False
                or not isinstance(fetch_order.get("order"), list)
                or any(type(i) is not int for i in fetch_order["order"])
                or sorted(fetch_order["order"]) != order
            ):
                raise ValueError("definition_fetch_order_binding")
            decisions = checked_fetch_decisions(fetch_order, hits[:limit])
            if not set(fetch_order["index_keys"]) <= set(self.index_keys):
                raise ValueError("definition_fetch_order_binding")
            order = fetch_order["order"]
        for offset, index in enumerate(order):
            hit = hits[index]
            if decisions is not None and decisions[index]["already_acquired_identity"]:
                # Defer this bounded attempt, not a claim that remote text is
                # unchanged. Current local material still uses original readers.
                backlog.append(
                    dict(
                        reason="known_source_fetch_deferred",
                        hit=hit,
                        decision=decisions[index],
                        remote_refresh_performed=False,
                        current_medical_qualification_granted=False,
                    )
                )
                continue

            def fetch_current(hit=hit):
                from packages.knowledge_connectors.ports import ConnectorError, ErrorCategory

                try:
                    return save_fetch(self.fetch_hit(hit), self.store)
                except DefinitionMaterialUnavailable as exc:
                    return dict(material_unavailable=str(exc), metadata=None)
                except ConnectorError as exc:
                    status = (
                        exc.metadata.attempts[-1].http_status
                        if exc.metadata and exc.metadata.attempts
                        else None
                    )
                    cause = (
                        exc.last_error
                        if exc.category == ErrorCategory.RETRY_EXHAUSTED
                        else exc.category
                    )
                    transient = cause in {
                        ErrorCategory.TIMEOUT,
                        ErrorCategory.NETWORK,
                        ErrorCategory.RATE_LIMIT,
                    } or (
                        cause == ErrorCategory.HTTP
                        and status is not None
                        and (status == 408 or 500 <= status <= 599)
                    )
                    if self.failure_policy == "technical_backlog_no_retry" and transient:
                        return dict(
                            technical_error=cause.value,
                            error_category=exc.category.value,
                            metadata=exc.metadata.model_dump(mode="json") if exc.metadata else None,
                            retried=False,
                        )
                    if exc.category not in {
                        ErrorCategory.UNSUPPORTED,
                        ErrorCategory.MIME,
                        ErrorCategory.PARSE,
                        ErrorCategory.RESPONSE_SIZE,
                    } and status not in {404, 410}:
                        raise  # Technical/unknown/security failures are not empty literature.
                    return dict(
                        material_unavailable=exc.category.value,
                        metadata=exc.metadata.model_dump(mode="json") if exc.metadata else None,
                    )

            # The injected fetch is the original connector -> FetchInput boundary.
            fetched = operation(key + f":fetch:{index}", "fetch_group", fetch_current)
            if fetched.get("status") == "public_budget_exhausted":
                backlog.append(
                    dict(
                        reason="definition_public_request_budget",
                        result=fetched,
                        unattempted_hits=[hits[i] for i in order[offset:]],
                    )
                )
                break
            if "technical_error" in fetched:
                backlog.append(
                    dict(
                        reason="source_fetch_technical_failure",
                        hit=hit,
                        result=fetched,
                        unattempted_hits=[hits[i] for i in order[offset + 1 :]],
                    )
                )
                break  # Retain obtained text; do not retry an unhealthy provider implicitly.
            if "material_unavailable" in fetched:
                backlog.append(dict(reason="source_material_unavailable", hit=hit, result=fetched))
                continue
            from db.knowledge.models import KnowledgeReference
            from packages.grounding.models import validate

            request = restore_fetch(fetched, self.store)
            if self.failure_policy == "technical_backlog_no_retry":
                # Explicit archive semantics, not a rewrite of the retained HTTP
                # receipt. The owner still verifies the persisted bounded page.
                request = request.model_copy(update={"pipeline_version": "1.2.0"})
            archived = self.repository.archive_hit(
                validate(KnowledgeReference, hit["reference"]),
                request,
                self.store,
            )
            options = self.index_options()

            def build(archived=archived, options=options):
                return self.repository.build_document_index(
                    archived,
                    self.store,
                    IndexConfig(
                        mode="fts_only",
                        vector_unavailable_reason="bounded_definition_text_retrieval",
                        vector_weight=0.0,
                    ),
                    applicability_refs=self.index_scope(archived),
                    **options,
                )

            if self.material_policy == "xml-jats-citations-windowed/1.0.0":
                indexes, notes = self.window_indexes(
                    request=request,
                    build=build,
                    options=options,
                    operation=operation,
                    key=key + f":fetch:{index}:index",
                )
                backlog.extend(dict(note, hit=hit) for note in notes)
                for indexed in indexes:
                    if indexed.get("status") != "indexed":
                        backlog.append(dict(reason="source_not_indexed", hit=hit, result=indexed))
                    elif indexed["key"] not in {i["key"] for i in acquired_indexes}:
                        acquired_indexes.append(indexed)
                # Select once over the complete window pool, not once per window.
                continue
            try:
                indexed = build()
            except ValueError as exc:
                if not self.material_policy or str(exc) != "table_chunk_limit":
                    raise
                from packages.knowledge_index.models import ProseChunkPolicy

                # Preserve the archived bytes and complete parse. Explicitly index
                # prose only, never split a table or pretend it was fully usable.
                options["chunk_policy"] = ProseChunkPolicy(
                    source_checksum=request.document.response.response_checksum,
                    max_chars=3000,
                    max_chunks=250,
                )
                indexed = build()
                backlog.append(
                    dict(
                        reason="table_excluded_from_prose_index",
                        hit=hit,
                        policy=options["chunk_policy"].model_dump(mode="json"),
                        index_key=indexed.get("key"),
                    )
                )
            if indexed.get("status") != "indexed":
                backlog.append(dict(reason="source_not_indexed", hit=hit, result=indexed))
                continue
            if self.intent_coverage:
                if indexed["key"] not in {i["key"] for i in acquired_indexes}:
                    acquired_indexes.append(indexed)
                # Cover all bounded obtained documents before selecting a quota.
                continue
            passage_limit = limit - len(found)
            if self.material_policy and not self.passage_plans:
                passage_limit = min(1, passage_limit)
            material = local_query((indexed["key"],), query, passage_limit)
            found.extend(material)
            if not material:
                backlog.append(
                    dict(reason="no_current_applicable_passage", hit=hit, index_key=indexed["key"])
                )
            if len(found) >= limit:
                break
        if (
            self.intent_coverage or self.material_policy == "xml-jats-citations-windowed/1.0.0"
        ) and acquired_indexes:
            found = local_query(tuple(i["key"] for i in acquired_indexes), query, limit)
            if not found:
                backlog.append(
                    dict(
                        reason="no_current_applicable_passage",
                        index_keys=[i["key"] for i in acquired_indexes],
                    )
                )
        return response(found, False, [*backlog, *self.query_backlog])

    def window_indexes(self, *, request, build, options, operation, key):
        """Freeze a complete bounded selection before original-owner writes.

        Confirmed per-window receipts belong to the campaign operation journal;
        they are never counted as separate source documents or HTTP requests.
        """
        from packages.knowledge_index.chunking import chunk, plan_windows
        from packages.knowledge_index.models import ProseChunkPolicy, parse_chunk_policy
        from packages.knowledge_index.parsers import parse

        response = request.document.response

        def plan():
            self.before_read()
            parsed = parse(
                response.body,
                response.mime,
                language=request.bibliography.language or "und",
                policy=options["parse_policy"],
            )
            if parsed.document is None:
                return dict(
                    policies=[],
                    reason=parsed.reason or parsed.status,
                    source_checksum=response.response_checksum,
                )
            selected = options["chunk_policy"]
            notes = []
            # Scan structural table capacity before count overflow can mask it.
            # No semantic filtering: the full source/parse still remains saved.
            if any(
                b.position.kind == "table" and len(b.text) > selected.max_chars
                for b in parsed.document.blocks
            ):
                selected = ProseChunkPolicy(
                    source_checksum=response.response_checksum,
                    max_chars=selected.max_chars,
                    overlap_chars=selected.overlap_chars,
                    max_chunks=selected.max_chunks,
                )
                notes.append(dict(reason="table_excluded_from_prose_index"))
            try:
                chunk(parsed.document, selected)
            except ValueError as error:
                if str(error) == "table_chunk_limit":
                    selected = ProseChunkPolicy(
                        source_checksum=response.response_checksum,
                        max_chars=selected.max_chars,
                        overlap_chars=selected.overlap_chars,
                        max_chunks=selected.max_chunks,
                    )
                    notes.append(dict(reason="table_excluded_from_prose_index"))
                elif str(error) != "chunk_count_limit":
                    raise
            else:
                # Small documents retain the original whole-document index key.
                return dict(
                    policies=[selected.model_dump(mode="json")],
                    source_checksum=response.response_checksum,
                    notes=notes,
                    original_blocks=len(parsed.document.blocks),
                )
            try:
                policies = plan_windows(
                    parsed.document,
                    source_checksum=response.response_checksum,
                    policy=selected,
                    max_windows=4,
                )
            except ValueError as error:
                if str(error) not in {
                    "source_window_limit",
                    "chunk_count_limit",
                    "prose_selection_empty",
                    "source_window_empty",
                }:
                    raise
                return dict(
                    policies=[], reason=str(error), source_checksum=response.response_checksum
                )
            return dict(
                policies=[p.model_dump(mode="json") for p in policies],
                source_checksum=response.response_checksum,
                notes=notes,
                original_blocks=len(parsed.document.blocks),
            )

        planned = operation(key + ":plan", "local_index_plan", plan)
        if planned["source_checksum"] != response.response_checksum:
            raise ValueError("definition_window_source_changed")
        if not planned["policies"]:
            return [], [dict(reason="source_index_capacity_or_parse_backlog", plan=planned)]
        if len(planned["policies"]) > 4:
            raise ValueError("source_window_limit")
        results = []
        for index, policy in enumerate(planned["policies"]):
            self.before_read()
            options["chunk_policy"] = parse_chunk_policy(policy)
            result = operation(key + f":window:{index}", "local_index_window", build)
            results.append(result)
            self.before_read()
        return results, planned["notes"]

    def index_options(self):
        if self.material_policy is None:
            return {}
        from packages.knowledge_index.models import ChunkPolicy, ParsePolicy

        return dict(
            parse_policy=ParsePolicy(
                version=(
                    "xml-jats-citations/1.0.0"
                    if self.material_policy
                    in {"xml-jats-citations-whole-text/1.0.0", "xml-jats-citations-windowed/1.0.0"}
                    else "xml-jats/1.0.0"
                    if self.material_policy == "xml-jats-whole-text/1.0.0"
                    else "xml-scripts/1.0.0"
                )
            ),
            chunk_policy=ChunkPolicy(max_chars=3000, max_chunks=250),
        )


class LiteratureDefinitionConnector:
    """Original registered Europe PMC search/fetch, zero internal retries.

    Only configured public vocabulary topics cross the existing privacy owner.
    Every wire attempt (including a redirect) reserves campaign capacity first.
    Construction is inert: no credentials, private config or network are read.
    """

    def __init__(self, repository, expansion, topics):
        self.repository, self.expansion, self.topics = repository, expansion, topics

    def governor(self, original=None, *, kind=None):
        connector = self
        from packages.knowledge_connectors.http import SystemClock

        class Governor:
            clock = original.clock if original else SystemClock()

            def remaining(self):
                from datetime import UTC, datetime

                from packages.knowledge_search.policy import SearchStopped
                from packages.llm_provider.provider import ProviderError

                try:
                    connector.expansion.fast_control()
                except ProviderError as exc:
                    reason = {"cancelled": "cancelled", "deadline": "timeout"}.get(exc.code)
                    if reason:
                        raise SearchStopped(reason) from exc
                    raise
                n = (
                    datetime.fromisoformat(connector.expansion.campaign.protocol["deadline"])
                    - datetime.now(UTC)
                ).total_seconds()
                return min(n, original.remaining()) if original else n

            def retry_limit(self, configured_limit):
                return 0

            prepared_provider = None

            def prepare_send(self, provider_code):
                from packages.llm_provider.provider import ProviderError

                if self.prepared_provider is not None:
                    raise ValueError("definition_wire_admission_already_prepared")
                e = connector.expansion
                try:
                    if hasattr(getattr(e, "policy", None), "max_search_requests"):
                        self.wire_key = e.reserve_wire(provider_code, kind=kind)
                    else:
                        self.wire_key = e.reserve_wire(provider_code)
                except ProviderError as exc:
                    raise DefinitionAdmissionStopped(exc) from exc
                self.prepared_provider = provider_code

            def before_send(self, provider_code):
                # Legacy callers still perform the same owner gate here. New
                # clients prepare it outside only the connector's wire timer.
                if self.prepared_provider is None:
                    self.prepare_send(provider_code)
                if self.prepared_provider != provider_code:
                    raise ValueError("definition_wire_admission_provider_changed")
                self.prepared_provider = None
                self.remaining()
                return original.before_send(provider_code) if original else self.remaining()

            def after_send(self, *args):
                if original:
                    original.after_send(*args)

            def completed_request(self, metadata):
                if original:
                    original.completed_request(metadata)

            def completed_wire(self, receipt):
                if original:
                    original.completed_wire(receipt)
                # Save confirmed local outcome before checking cancellation on return.
                e = connector.expansion
                with e.documents.session():
                    e.documents.write(self.wire_key + ":outcome", (receipt,))
                from packages.knowledge_search.policy import SearchStopped
                from packages.llm_provider.provider import ProviderError

                try:
                    e.control()
                except ProviderError as exc:
                    reason = {"cancelled": "cancelled", "deadline": "timeout"}.get(exc.code)
                    if reason:
                        raise SearchStopped(reason) from exc
                    raise DefinitionAdmissionStopped(exc) from exc

            def wait(self, seconds):
                connector.expansion.fast_control()
                if original:
                    original.wait(seconds)
                else:
                    self.clock.sleep(min(seconds, self.remaining()))

        return Governor()

    def search_target(self):
        """Original public vocabulary binding; narrow owners may project a root."""
        policy = getattr(self.expansion, "policy", None)
        target = getattr(policy, "public_target", None)
        if target is not None and target != self.expansion.binding["target"]:
            raise ValueError("definition_public_target_binding")
        return target

    def search(self, *, query, key, limit):
        from apps.knowledge_literature_tools import search

        if query not in self.topics:
            raise ValueError("definition_public_topic_not_configured")
        selected = self.topics[query]
        combined = not isinstance(selected, str)
        target = self.search_target()
        result = search(
            self.repository,
            ("europepmc",),
            "glaucoma" if combined else selected,
            limit=limit,
            key=key,
            request_guard=self.governor(kind="search"),
            timeout_ms=getattr(getattr(self.expansion, "policy", None), "search_timeout_ms", 30000),
            **(dict(topics=tuple(selected)) if combined else {}),
            **(dict(public_target=target) if target is not None else {}),
            **(
                dict(public_target_vocabulary=self.expansion.policy.public_target_vocabulary)
                if getattr(
                    getattr(self.expansion, "policy", None), "public_target_vocabulary", None
                )
                is not None
                else {}
            ),
        )
        policy = getattr(self.expansion, "policy", None)
        outcomes = [r["outcome"] for r in result.get("runs", [])]
        bounded_partial = (
            getattr(policy, "search_failure_policy", None) == "technical_backlog_no_retry"
            and result["status"] == "partial"
            and bool(outcomes)
            and all(
                o.get("stop_reason") in {"complete", "empty", "page_limit", "hit_budget"}
                and o.get("error_category") is None
                for o in outcomes
            )
        )
        if result["status"] not in {"succeeded", "empty"} and not bounded_partial:
            if (
                getattr(policy, "search_failure_policy", None) == "technical_backlog_no_retry"
                and result["status"] in {"timeout", "failed", "partial"}
                and not any(
                    o.get("stop_reason") == "cancelled"
                    or o.get("error_category") in {"security", "authentication"}
                    for o in outcomes
                )
            ):
                return dict(
                    status="technical_backlog",
                    reason="public_search_incomplete",
                    query=query,
                    request_key=key,
                    result=result,
                    retried=False,
                )
            # Original Knowledge retains the detailed outcomes. A failed search is
            # not "no literature"; its campaign reservation remains non-repeatable.
            raise ValueError("definition_public_search_not_complete:" + result["status"])
        return [h for group in result["results"] for h in group["hits"]][:limit]

    def fetch(self, hit):
        import re

        from apps.knowledge_literature_tools import active_provider, client
        from db.knowledge.models import KnowledgeReference, KnowledgeStatus
        from packages.grounding.models import validate
        from packages.knowledge_archive import FetchInput
        from packages.knowledge_connectors import SourceLocator
        from packages.knowledge_connectors.configuration import ConnectorConfiguration
        from packages.knowledge_connectors.registry import ProviderRegistry

        current = self.repository.get(validate(KnowledgeReference, hit["reference"]))
        if current.withdrawn or current.status == KnowledgeStatus.DEPRECATED:
            raise ValueError("definition_search_hit_unavailable")
        record = current.record
        # Search record IDs (e.g. MED:123 or PMC:PMC123) are not full-text locators.
        # Use only the typed identity on the persisted hit, which the archive owner
        # independently checks against the returned document's locator.
        pmcids = {
            i.value for i in record.discovery.discovery_identifiers if i.identifier_type == "PMCID"
        }
        if not pmcids:
            raise DefinitionMaterialUnavailable("europepmc_full_text_pmcid_missing")
        if len(pmcids) != 1 or not re.fullmatch(r"PMC[1-9][0-9]*", next(iter(pmcids))):
            raise ValueError("definition_full_text_pmcid_identity")
        config, _ = active_provider(self.repository, "europepmc")
        registry = ProviderRegistry(
            ConnectorConfiguration(config_version="1.0.0", providers=(config,)),
            lambda c: client(c, self.governor(kind="fetch")),
        )
        try:
            document = registry.get(config.provider_code).fetch(
                SourceLocator(record_id=next(iter(pmcids)))
            )
        except DefinitionAdmissionStopped as exc:
            raise exc.original from exc
        return FetchInput(
            document=document, metadata_response_checksum=document.response.response_checksum
        )
