"""Explicit provider ports and deterministic synthetic fixtures, never remote APIs."""

import math
from typing import Protocol

from .models import ProviderSpec


class Embedder(Protocol):
    spec: ProviderSpec

    def embed(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]: ...


class Reranker(Protocol):
    spec: ProviderSpec

    def score(self, query: str, texts: tuple[str, ...]) -> tuple[float, ...]: ...


def validate_vectors(values, count: int, dimension: int):
    if len(values) != count:
        raise ValueError("embedding count mismatch")
    for row in values:
        if len(row) != dimension or not all(
            type(v) in (int, float) and math.isfinite(v) for v in row
        ):
            raise ValueError("embedding dimension or finite-value mismatch")
    return values


class FakeEmbedder:
    """Only a synthetic evaluation fixture; no semantic quality assertion."""

    def __init__(self, dimension=3):
        self.spec = ProviderSpec(
            provider="offline-fake",
            config_version="1.0.0",
            model="fixture-terms",
            model_version="1.0.0",
            dimension=dimension,
        )

    def embed(self, texts):
        terms = ("glaucoma", "retina", "negative")
        return tuple(
            tuple(float(text.lower().count(terms[i % 3])) for i in range(self.spec.dimension))
            for text in texts
        )


class FakeReranker:
    def __init__(self):
        self.spec = ProviderSpec(
            provider="offline-fake",
            config_version="1.0.0",
            model="fixture-overlap",
            model_version="1.0.0",
            dimension=1,
        )

    def score(self, query, texts):
        terms = set(query.lower().split())
        return tuple(float(len(terms & set(text.lower().split()))) for text in texts)
