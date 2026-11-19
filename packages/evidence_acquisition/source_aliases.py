"""Exact local research-source identities; never fuzzy-repair a model response.

The installed owner rebuilds these from current authorized sources. Aliases are
bound by both the Material checksum and a persisted reversible provenance map.
Original Knowledge objects and fixed model sources are never changed.
"""

from packages.evidence_acquisition.materials import _rebuild
from packages.grounding.models import digest, validate
from packages.llm_provider.models import TaskDefinition


def alias_sources(sources, *, previous=()):
    if len({s.reference.code for s in sources}) != len(sources):
        raise ValueError("source_alias_duplicate")
    if any(s.state != "research_only" or s.release is not None for s in sources):
        raise ValueError("source_alias_research_only")
    # Prior names are not authorization: each must reconstruct exactly from a
    # currently owner-readable original source. The accepted reader separately
    # verifies the old submission, receipt and full visible context.
    import re

    reserved, prior = set(), {}
    for old in previous:
        code = old.reference.code
        if (
            not re.fullmatch(r"SRC:[0-9]{2,}", code)
            or old.reference.version != "research-source-alias/1.0.0"
            or code in reserved
            or old.reference.checksum in prior
        ):
            raise ValueError("source_alias_prior_identity")
        reserved.add(code)
        prior[old.reference.checksum] = old
    current = {digest(s): s for s in sources}
    if not set(prior) <= current.keys():
        raise ValueError("source_alias_prior_material_missing_or_changed")
    replacements, bindings, next_index = {}, [], 1
    for source in sorted(sources, key=lambda s: s.reference.code):
        old = prior.get(digest(source))
        if old is None:
            while f"SRC:{next_index:02d}" in reserved:
                next_index += 1
            code = f"SRC:{next_index:02d}"
            reserved.add(code)
        else:
            code = old.reference.code
        reference = source.reference.model_copy(
            update={
                "code": code,
                "version": "research-source-alias/1.0.0",
                "checksum": digest(source),
            }
        )
        projected = source.model_copy(update={"reference": reference})
        if old is not None and projected != old:
            raise ValueError("source_alias_prior_material_changed")
        replacements[source.reference.code] = projected
        bindings.append(
            dict(
                alias=reference.model_dump(mode="json"),
                original=source.reference.model_dump(mode="json"),
                source_checksum=digest(source),
            )
        )
    manifest = dict(
        schema_version="research-source-aliases/1.1.0"
        if previous
        else "research-source-aliases/1.0.0",
        bindings=bindings,
    )
    if previous:
        manifest["prior_sources_checksum"] = digest(previous)
    return tuple(replacements[s.reference.code] for s in sources), dict(
        **manifest, checksum=digest(manifest)
    )


def prepare_source_aliases(state):
    fixed = {code for p in state.policy.items if p.fixed_model_row for code in p.source_codes}
    original = tuple(s for s in state.material.sources if s.reference.code not in fixed)
    aliases, manifest = alias_sources(original)
    mapping = {old.reference.code: new for old, new in zip(original, aliases, strict=True)}
    items = []
    for item in state.policy.items:
        if item.fixed_model_row is not None:
            items.append(item)
            continue
        body = item.definition.model_dump(mode="json", exclude={"reference"})
        body["source_refs"] = [
            mapping[r.code].reference.model_dump(mode="json")
            if r.code in mapping
            else r.model_dump(mode="json")
            for r in item.definition.source_refs
        ]
        body["reference"] = item.definition.reference.model_copy(
            update={"version": "research-source-alias/1.0.0", "checksum": digest(body)}
        ).model_dump(mode="json")
        items.append(
            item.model_copy(
                update={
                    "source_codes": tuple(
                        mapping[c].reference.code if c in mapping else c for c in item.source_codes
                    ),
                    "definition": validate(TaskDefinition, body),
                }
            )
        )
    return _rebuild(
        state,
        items=tuple(items),
        sources=tuple(mapping.get(s.reference.code, s) for s in state.material.sources),
    ), manifest
