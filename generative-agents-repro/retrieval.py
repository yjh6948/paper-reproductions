import math
from collections import Counter
from typing import List, Tuple

from memory import MemoryObject


def recency_score(memory: MemoryObject, now: float, decay: float = 0.99) -> float:
    hours_since_access = max(0.0, now - memory.last_accessed_at)
    return decay ** hours_since_access


def _tokenize(text: str) -> list:
    return [w.lower() for w in text.replace(",", " ").replace(".", " ").split()]


def _cosine(a: Counter, b: Counter) -> float:
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def relevance_score(memory: MemoryObject, query: str) -> float:
    return _cosine(Counter(_tokenize(memory.content)), Counter(_tokenize(query)))


def _min_max_normalize(values: list) -> list:
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def retrieve_top_k(
    memories: List[MemoryObject], query: str, now: float, k: int = 3, decay: float = 0.99
) -> List[Tuple[MemoryObject, float]]:
    recencies = [recency_score(m, now, decay) for m in memories]
    importances = [m.importance or 0.0 for m in memories]
    relevances = [relevance_score(m, query) for m in memories]

    norm_rec = _min_max_normalize(recencies)
    norm_imp = _min_max_normalize(importances)
    norm_rel = _min_max_normalize(relevances)

    scored = list(zip(memories, (r + i + v for r, i, v in zip(norm_rec, norm_imp, norm_rel))))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:k]
