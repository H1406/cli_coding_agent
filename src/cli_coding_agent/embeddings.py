from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod


class EmbeddingClient(ABC):
    model_name: str
    version: str

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_many_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]


class HashEmbeddingClient(EmbeddingClient):
    def __init__(
        self,
        model_name: str = "hash-embedding",
        version: str = "v1",
        dimensions: int = 32,
    ) -> None:
        self.model_name = model_name
        self.version = version
        self.dimensions = dimensions

    def embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = digest[0] % self.dimensions
            sign = 1.0 if digest[1] % 2 == 0 else -1.0
            magnitude = (digest[2] / 255.0) + 0.5
            vector[index] += sign * magnitude
        return _normalize(vector)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Vectors must have the same number of dimensions.")
    numerator = sum(lhs * rhs for lhs, rhs in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]
