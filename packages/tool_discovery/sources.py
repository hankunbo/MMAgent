"""Bounded installed synthetic metadata only; no transport, imports or execution."""

from packages.grounding.models import canonical, digest, validate

from .models import SOURCE_ORDER, RegisteredIdentity
from .versions import parse


class SyntheticSource:
    def __init__(self, name, kind, records=(), *, identities=(), technical_failure=False):
        if kind not in SOURCE_ORDER or type(technical_failure) is not bool:
            raise ValueError("discovery_source_kind")
        self.name, self.kind = name, kind
        self._records = tuple(parse("Provenance", r) for r in records)
        self._identities = tuple(validate(RegisteredIdentity, r) for r in identities)
        if any(r.source_kind != kind for r in self._records):
            raise ValueError("discovery_source_binding")
        if len(self._records) > 128 or len(canonical(self._records)) > 524288:
            raise ValueError("discovery_source_bound")
        if len({r.identity_ref for r in self._identities}) != len(self._identities):
            raise ValueError("discovery_registry_identity_conflict")
        if len(self._identities) > 128 or len(canonical(self._identities)) > 524288:
            raise ValueError("discovery_source_bound")
        self.technical_failure = technical_failure

    @property
    def fingerprint(self):
        return digest(
            dict(
                name=self.name,
                kind=self.kind,
                records=self._records,
                identities=self._identities,
                technical_failure=self.technical_failure,
            )
        )

    def read(self, query, *, limit):
        if not 1 <= limit <= 128:
            raise ValueError("discovery_source_bound")
        if self.technical_failure:
            raise OSError("synthetic_candidate_source_unavailable")
        return self._records[:limit], len(self._records) > limit

    def identity(self, ref):
        return next((r for r in self._identities if r.identity_ref == ref), None)
