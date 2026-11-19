"""Public deterministic Writer; never starts/retries/resolves an original run."""

from packages.grounding.models import Token, digest
from packages.memory.writer_models import EpisodeData, Segment
from packages.memory_writer.projection import NotReady, project, terminal, validate_aggregate


class WriterUnavailable(RuntimeError):
    """Sanitized unknown owner failure; never a withdrawal or medical outcome."""


class MemoryWriter:
    def __init__(self, *, repository, sources):
        self.repository, self.sources = repository, sources

    def prepare(self, *, run_id, request_key, supplemental=()):
        from pydantic import TypeAdapter

        TypeAdapter(Token).validate_python(request_key)
        if run_id != self.sources.run_id:
            raise ValueError("memory_scope")
        if len(supplemental) > 16 or len(set(supplemental)) != len(supplemental):
            raise ValueError("memory_supplement_bound")
        snapshot = self.sources.snapshot()
        source = self.sources.snapshot_reference(snapshot)
        terminal(snapshot, source)
        try:
            facts = self.sources.required_fact_sources(snapshot)
        except ValueError as error:
            if type(error) is ValueError and str(error) == "memory_fact_reader_required":
                raise NotReady("memory_fact_reader_required") from None
            raise
        extras = tuple(self.sources.reference(*item) for item in supplemental)
        return project(
            snapshot, source, request_key=request_key, fact_sources=facts, supplements=extras
        )

    def write(self, *, run_id, request_key, supplemental=()):
        try:
            records = self.prepare(
                run_id=run_id, request_key=request_key, supplemental=supplemental
            )
        except NotReady as error:
            return dict(
                status="not_ready",
                reason=str(error),
                run_id=run_id,
                saved=False,
                live_domain_acceptance="live_domain_acceptance_not_enabled",
            )
        except (OSError, RuntimeError) as error:
            from packages.evidence_details.current_rejection import current_rejection

            if current_rejection(error):
                raise
            raise WriterUnavailable("memory_writer_source_unavailable") from None
        result = self.repository.create_episode(request_key, records)
        replay = self.replay(result.record.key)
        if replay["episode_checksum"] != result.checksum:
            raise ValueError("memory_read_changed")
        return result

    def replay(self, key, *, historical=False):
        """Read saved ordered summaries and bindings; no owner execution, simulation or resolve."""
        result = self.repository.read(key, historical=historical)
        episode = result.record
        if not isinstance(getattr(episode, "detail", None), EpisodeData):
            raise ValueError("memory_writer_episode_required")
        members = tuple(
            self.repository.read(p.key, historical=historical).record for p in episode.parents
        )
        validate_aggregate(episode.detail.request_key, (*members, episode))
        by_key = {m.key: m for m in members}
        steps = tuple(
            s
            for k in episode.detail.segments
            for s in by_key[k].detail.steps
            if isinstance(by_key[k].detail, Segment)
        )
        return dict(
            schema_version="memory-writer-replay/1.0.0",
            episode=key,
            episode_checksum=digest(episode),
            current=not historical,
            terminal=episode.detail.terminal.model_dump(mode="json"),
            steps=[s.model_dump(mode="json") for s in steps],
            eligible_for_retrieval=False,
            eligible_for_planning=False,
        )
