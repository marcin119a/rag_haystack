
from __future__ import annotations

import math


def reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    for position, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / position
    return 0.0


def _dcg(gains: list[int]) -> float:
    return sum((2**gain - 1) / math.log2(position + 1) for position, gain in enumerate(gains, start=1))


def ndcg_at_k(ranked_ids: list[str], relevance: dict[str, int], k: int) -> float:
    dcg = _dcg([relevance.get(doc_id, 0) for doc_id in ranked_ids[:k]])
    idcg = _dcg(sorted(relevance.values(), reverse=True)[:k])
    return dcg / idcg if idcg > 0 else 0.0
