
from __future__ import annotations

import time
from dataclasses import dataclass, field
from statistics import mean, median

from search.base import Searcher
from search.eval.qrels import QUERIES, Query


@dataclass(frozen=True)
class BenchResult:
    variant: str
    n_queries: int
    repeats: int
    latencies_s: list[float] = field(repr=False)

    @property
    def mean_s(self) -> float:
        return mean(self.latencies_s)

    @property
    def median_s(self) -> float:
        return median(self.latencies_s)

    @property
    def p95_s(self) -> float:
        ordered = sorted(self.latencies_s)
        return ordered[max(0, round(0.95 * (len(ordered) - 1)))]

    @property
    def min_s(self) -> float:
        return min(self.latencies_s)

    @property
    def max_s(self) -> float:
        return max(self.latencies_s)


def bench(
    variant_name: str,
    searcher: Searcher,
    queries: list[Query] = QUERIES,
    repeats: int = 3,
    warmup: int = 1,
) -> BenchResult:
    texts = [q.text for q in queries]

    for _ in range(warmup):
        for text in texts:
            searcher.search(text)

    latencies: list[float] = []
    for _ in range(repeats):
        for text in texts:
            start = time.perf_counter()
            searcher.search(text)
            latencies.append(time.perf_counter() - start)

    return BenchResult(variant=variant_name, n_queries=len(texts), repeats=repeats, latencies_s=latencies)
