"""Versioned Agent documents through original CASE Artifact/Runtime/Audit ports.

No new SQL, run system, overwrite, or clinical graph acceptance. Raw prompts/responses
are not journaled; domain candidates contain approved visible values only.
"""

import json
from contextlib import contextmanager
from datetime import UTC, datetime

from db.runtime.repository.case_pipeline import identity
from packages.contracts import Artifact
from packages.grounding.models import canonical, digest
from packages.object_store import checksum


class RuntimeDocuments:
    def __init__(self, *, repo, contexts, sessions, store, run_id, root):
        self.repo, self.contexts, self.sessions, self.store = repo, contexts, sessions, store
        self.run_id, self.root = run_id, root
        self._check_root()

    def _check_root(self):
        if not hasattr(self.repo, "read_case_artifact"):
            self._check()
            return
        context = self.contexts.read_research_run(self.run_id)
        case = context.request.scope.case_code
        if (
            self.root.scope.case_code != case
            or self.repo.read_case_artifact(case, self.root.asset_code) != self.root
        ):
            raise ValueError("agent_runtime_scope")
        self.read_artifact(self.root)

    def _check(self):
        context = self.contexts.read_research_run(self.run_id)
        bundle = self.repo.get_case_bundle(context.request.scope.case_code)
        saved = next((a for a in bundle.artifacts if a.asset_code == self.root.asset_code), None)
        if saved != self.root or self.root.scope.case_code != context.request.scope.case_code:
            raise ValueError("agent_runtime_scope")
        self.read_artifact(self.root)
        return bundle

    @contextmanager
    def session(self):
        # Distinct from Planner's existing control lock: Trace can journal under select.
        with self.sessions.session(self.run_id + ":agent-documents"):
            yield

    def read_artifact(self, artifact):
        raw = self.store.read_bounded(artifact.uri, 4_194_304)
        if checksum(raw) != artifact.checksum:
            raise ValueError("agent_object_checksum")
        return raw

    def key(self, key):
        return identity("TASK", self.run_id + ":agent:" + key)

    def validate_scope(self, scope):
        if scope.run_id != self.run_id or scope.case_code != self.root.scope.case_code:
            raise ValueError("agent_runtime_scope")
        if hasattr(self.repo, "read_case_artifact"):
            self._check_root()
            graph = self.repo.read_case_artifact(scope.case_code, scope.graph.code)
        else:
            graph = next(
                (a for a in self._check().artifacts if a.asset_code == scope.graph.code), None
            )
        if graph is None or graph.checksum != scope.graph.checksum:
            raise ValueError("agent_runtime_graph")
        value = json.loads(self.read_artifact(graph))
        if graph.asset_code != self.root.asset_code:
            if (
                value.get("run_id") != self.run_id
                or value.get("root_checksum") != self.root.checksum
            ):
                raise ValueError("agent_runtime_graph_lineage")
            value = value["value"]
        if value.get("schema_version", value.get("schema")) != scope.graph.version:
            raise ValueError("agent_runtime_graph_version")

    def validate_snapshot(self, *, case_code, run_id, snapshot):
        """Bind a formal snapshot via its original repository, not an Artifact cast.

        This is storage identity only; current source admission belongs to the
        installed domain reader, and is deliberately absent on historical reads.
        """
        from db.runtime.repository.ports import NotFoundError
        from packages.rule_engine import snapshot_binding_for

        if run_id != self.run_id or case_code != self.root.scope.case_code:
            raise ValueError("agent_runtime_scope")
        self._check()
        try:
            graph = self.repo.get_graph_snapshot(snapshot.graph_code, snapshot.graph_version)
        except NotFoundError as error:
            raise ValueError("agent_runtime_graph") from error
        if (
            graph.case_code != case_code
            or graph.scope.case_code != case_code
            or graph.scope.run_id != run_id
            or snapshot_binding_for(graph) != snapshot
        ):
            raise ValueError("agent_runtime_graph")

    def failure(self, key, request_checksum, reason):
        return self.repo.record_agent_failure(
            case_code=self.root.scope.case_code,
            run_id=self.run_id,
            key=key,
            request_checksum=request_checksum,
            reason=reason,
        )

    def envelope(self, key, index, value):
        return dict(
            schema_version="agent-runtime-document/1.0.0",
            run_id=self.run_id,
            root_checksum=self.root.checksum,
            key_checksum=digest(key),
            index=index,
            value=json.loads(canonical(value)),
        )

    def reference(self, key, index, value, *, role, version):
        from packages.planner_control.models import Reference

        sha = checksum(canonical(self.envelope(key, index, value)))
        return Reference(
            role=role, code=identity("ASSET", self.run_id + sha), version=version, checksum=sha
        )

    def _known_consumption(self, graph):
        """Fresh exact receipt lookup; a phase is a locator, never authorization.

        Unsupported/legacy locators fall back to the complete original inventory.
        The positive path proves current CASE membership, original object bytes,
        run/root/key identity and the same accepted parent graph as that scan.
        No result, source qualification or metadata is cached between calls.
        """
        repo = getattr(self, "repo", None)
        if not hasattr(repo, "read_case_artifact") or not hasattr(repo, "read_observation_bundle"):
            return False
        self._check_root()
        phase = self.read("dynamic-phase-result")
        if not phase or len(phase) != 1 or not isinstance(phase[0], dict):
            return False
        value = phase[0]
        key = value.get("acceptance_key")
        if (
            value.get("run_id") != self.run_id
            or value.get("accepted") is not True
            or not isinstance(key, str)
            or not 1 <= len(key) <= 256
        ):
            return False
        name = "accepted:" + key
        bundle = self.repo.read_observation_bundle(self.key(name))
        if not bundle:
            return False
        # Original acceptance commits its receipt as the last bundle artifact.
        # Earlier/other layouts still take the old complete scan below.
        artifact = bundle[-1]
        if (
            artifact.artifact_type_code != "ARTIFACT:AGENT_RECORD"
            or artifact.scope != self.root.scope
            or self.repo.read_case_artifact(self.root.scope.case_code, artifact.asset_code)
            != artifact
        ):
            return False
        envelope = json.loads(self.read_artifact(artifact))
        if (
            envelope.get("run_id") != self.run_id
            or envelope.get("root_checksum") != self.root.checksum
            or envelope.get("key_checksum") != digest(name)
            or envelope.get("index") != len(bundle) - 1
        ):
            return False
        receipt = envelope.get("value")
        return (
            isinstance(receipt, dict)
            and receipt.get("schema_version")
            in {
                "research-acceptance/1.0.0",
                "research-acceptance/2.0.0",
                "research-acceptance/3.0.0",
            }
            and receipt.get("status") == "accepted"
            and isinstance(receipt.get("parent_graph"), dict)
            and receipt["parent_graph"].get("code") == graph.code
            and receipt["parent_graph"].get("checksum") == graph.checksum
        )

    def consumed(self, graph):
        if self._known_consumption(graph):
            return True
        # Writer identities offer an ordering hint, never authorization or an
        # exclusion filter. Current metadata and every examined object's bytes
        # still pass the original checks; noncanonical/legacy records remain in
        # the fallback inventory. No accepted result is cached between calls.
        records = [
            a for a in self._check().artifacts if a.artifact_type_code == "ARTIFACT:AGENT_RECORD"
        ]
        records.sort(key=lambda a: a.asset_code != identity("ASSET", self.run_id + a.checksum))
        for a in records:
            d = json.loads(self.read_artifact(a))
            if d.get("run_id") != self.run_id or d.get("root_checksum") != self.root.checksum:
                continue
            v = d["value"]
            if (
                v.get("schema_version")
                in {
                    "research-acceptance/1.0.0",
                    "research-acceptance/2.0.0",
                    "research-acceptance/3.0.0",
                }
                and v.get("status") == "accepted"
            ):
                if (
                    v["parent_graph"]["code"] == graph.code
                    and v["parent_graph"]["checksum"] == graph.checksum
                ):
                    return True
        return False

    def read(self, key):
        prior = self.repo.read_observation_bundle(self.key(key))
        if prior is None:
            return None
        docs = tuple(json.loads(self.read_artifact(a)) for a in prior)
        if any(
            d["run_id"] != self.run_id or d["root_checksum"] != self.root.checksum for d in docs
        ):
            raise ValueError("agent_document_scope")
        return tuple(d["value"] for d in docs)

    def read_many(self, keys):
        """Fresh owner reads; exact same scope/object validation as read()."""
        bundles = self.repo.read_observation_bundles(tuple(self.key(k) for k in keys))
        result = []
        for bundle in bundles:
            if bundle is None:
                result.append(None)
                continue
            docs = tuple(json.loads(self.read_artifact(a)) for a in bundle)
            if any(
                d["run_id"] != self.run_id or d["root_checksum"] != self.root.checksum for d in docs
            ):
                raise ValueError("agent_document_scope")
            result.append(tuple(d["value"] for d in docs))
        return tuple(result)

    def write(self, key, values):
        """Caller holds session for read-check-write. Immutable atomic artifact bundle."""
        self._check_root()
        prior = self.read(key)
        expected = tuple(json.loads(canonical(v)) for v in values)
        if prior is not None:
            if prior != expected:
                raise ValueError("agent_idempotency_conflict")
            return self.repo.read_observation_bundle(self.key(key))
        outputs = []
        for index, value in enumerate(expected):
            envelope = self.envelope(key, index, value)
            raw = canonical(envelope)
            if len(raw) > 4_194_304:
                raise ValueError("agent_object_bound")
            sha = checksum(raw)
            stage = self.store.stage(raw)
            try:
                uri = self.store.promote(stage, sha)
            except BaseException:
                self.store.discard(stage)
                raise
            a = self.root.model_dump(mode="json")
            a.update(
                asset_code=identity("ASSET", self.run_id + sha),
                artifact_type_code="ARTIFACT:AGENT_RECORD",
                uri=uri,
                checksum=sha,
                media_type="application/json",
                derived_from=[self.root.asset_code],
                dimensions=[],
                measurements=[],
                research_only=True,
            )
            a["provenance"].update(
                generator_version="agent-record-1.0.0",
                created_at=datetime.now(UTC).isoformat(),
                source_codes=[self.root.asset_code],
                input_checksums=[self.root.checksum],
            )
            outputs.append(Artifact.model_validate(a))
        # Uncertain commit: retain immutable objects, retry reads the original receipt.
        self.repo.register_observation_bundle(tuple(outputs), self.key(key))
        return tuple(outputs)


class RuntimeAgentJournal:
    """Append-only ordered request journal, including durable pre-send reservations."""

    durable = True

    def __init__(self, documents):
        self.documents = documents

    def session(self):
        return self.documents.session()

    def validate_binding(self, run_id, deadline):
        if run_id != self.documents.run_id:
            raise ValueError("agent_journal_run_scope")
        start, _ = self.documents.sessions.origin(run_id)
        if deadline <= start:
            raise ValueError("agent_journal_deadline")

    def read(self):
        events = []
        # Historical acceptance uses one exact private proof-tracking wrapper.
        # Preserve original Runtime batching through it rather than opening a
        # new connection for every journal entry. No caller capability flag and
        # no new document/qualification cache is introduced.
        from packages.evidence_details.accepted_read_scope import _Documents

        owner = self.documents.owner if type(self.documents) is _Documents else self.documents
        if isinstance(owner, RuntimeDocuments) and hasattr(owner.repo, "read_observation_bundles"):
            for start in range(0, 1000, 64):
                keys = tuple("journal:" + str(n) for n in range(start, min(start + 64, 1000)))
                for values in self.documents.read_many(keys):
                    if values is None:
                        return tuple(events)
                    events.append(values[0])
            raise ValueError("agent_journal_limit")
        for n in range(1000):
            values = self.documents.read("journal:" + str(n))
            if values is None:
                return tuple(events)
            events.append(values[0])
        raise ValueError("agent_journal_limit")

    def append(self, event):
        self.documents.write("journal:" + str(len(self.read())), (event,))
