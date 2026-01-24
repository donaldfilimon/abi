"""
ABI Framework Embeddings Module

Provides lightweight embedding utilities with deterministic vectors
for development and testing workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional
import hashlib
import math
import random
import time

from .config import EmbeddingsConfig


@dataclass
class EmbeddingResult:
    """Result of an embedding operation."""

    text: str
    vector: List[float]
    model: str
    dimension: int
    normalized: bool
    timestamp: float


def _seed_from_text(text: str, model: str) -> int:
    digest = hashlib.sha256(f"{model}:{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


class EmbeddingsModel:
    """
    Deterministic embedding model for local usage.

    This implementation provides stable vectors without requiring
    external model files.
    """

    def __init__(self, config: Optional[EmbeddingsConfig] = None):
        self.config = config or EmbeddingsConfig.defaults()

    @property
    def dimension(self) -> int:
        return self.config.dimension

    def embed(self, text: str) -> EmbeddingResult:
        """Embed a single text string into a vector."""
        rng = random.Random(_seed_from_text(text, self.config.model))
        vector = [rng.uniform(-1.0, 1.0) for _ in range(self.config.dimension)]
        if self.config.normalize:
            vector = _normalize(vector)
        return EmbeddingResult(
            text=text,
            vector=vector,
            model=self.config.model,
            dimension=self.config.dimension,
            normalized=self.config.normalize,
            timestamp=time.time(),
        )

    def embed_batch(self, texts: Iterable[str]) -> List[EmbeddingResult]:
        """Embed multiple texts and return ordered results."""
        return [self.embed(text) for text in texts]

    def similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b):
            raise ValueError("Vectors must have the same length")
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


def embed_texts(
    texts: Iterable[str], config: Optional[EmbeddingsConfig] = None
) -> List[EmbeddingResult]:
    """Convenience helper for one-off embeddings."""
    model = EmbeddingsModel(config)
    return model.embed_batch(texts)
