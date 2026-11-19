"""04D combined pagination/order and deterministic Unicode position probes."""

import random

import pytest

from packages.knowledge_index.chunking import chunk
from packages.knowledge_index.models import ChunkPolicy, canonical
from packages.knowledge_index.parsers import parse
from tests.knowledge_search.support import assembly, config, exchange, plan


@pytest.mark.parametrize("seed", [4, 17, 65])
def test_provider_order_page_retry_and_duplicate_discovery(seed):
    rng = random.Random(seed)
    configs = [config(provider_code=f"SOURCE:PROVIDER:D04_{i}") for i in range(3)]
    expected = None
    for ordered in (configs, list(reversed(configs))):
        query = plan(*ordered)
        tapes = {}
        for cfg in configs:
            ids = ["81001", "81002"]
            rng.shuffle(ids)
            tapes[cfg.provider_code] = (
                exchange(cfg, query, status=503),
                exchange(cfg, query, ids=tuple(ids), next_cursor="second"),
                exchange(cfg, query, ids=(ids[0], "81003"), cursor="second"),
            )
        executor, transports = assembly(tuple(ordered), tapes)
        with executor:
            result = executor.execute(query)
        assert result.status == "succeeded"
        actual = {
            p.query.identity.provider_code: {h.metadata.provider_record_id for h in p.hits}
            for p in result.providers
        }
        assert all(values == {"MED:81001", "MED:81002", "MED:81003"} for values in actual.values())
        assert all(p.wire_requests == 3 and p.hit_count == 3 for p in result.providers)
        for transport in transports.values():
            transport.assert_consumed()
        if expected is not None:
            assert actual == expected
        expected = actual


@pytest.mark.parametrize("seed", [104, 204, 304])
def test_random_legal_paragraphs_keep_exact_unicode_positions(seed):
    rng = random.Random(seed)
    paragraphs = [
        " ".join(rng.choices(["OCT", "合成", "retina", "不确定", "1.5 mm", "αβ"], k=40)) + "."
        for _ in range(rng.randint(2, 5))
    ]
    data = ("<article>" + "".join(f"<p>{p}</p>" for p in paragraphs) + "</article>").encode()
    parsed = parse(data, "application/xml")
    repeated = parse(data, "application/xml")
    assert parsed.status == "parsed" and canonical(parsed) == canonical(repeated)
    assert [block.text for block in parsed.document.blocks] == paragraphs
    policy = ChunkPolicy(max_chars=128, overlap_chars=17)
    chunks = chunk(parsed.document, policy)
    assert chunks == chunk(repeated.document, policy)
    text = "\n".join(paragraphs)
    covered = set()
    for part in chunks:
        start, end = part.position.span_start, part.position.span_end
        assert text[start:end] == part.text and 0 < end - start <= policy.max_chars
        covered.update(range(start, end))
    assert all(i in covered for i, char in enumerate(text) if char != "\n")
