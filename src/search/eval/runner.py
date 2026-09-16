
from dataclasses import dataclass
from statistics import mean

from search.base import Searcher
from search.eval.metrics import ndcg_at_k, reciprocal_rank
from search.eval.qrels import Query


@dataclass(frozen=True)
class QueryResult:
    query: str
    ranked: list[str]
    reciprocal_rank: float
    ndcg: float


@dataclass(frozen=True)
class EvalResult:
    variant: str
    k: int
    per_query: list[QueryResult]

    @property
    def mrr(self) -> float:
        return mean(r.reciprocal_rank for r in self.per_query)

    @property
    def mean_ndcg(self) -> float:
        return mean(r.ndcg for r in self.per_query)


def _ranked_course_names(docs) -> list[str]:
    seen: list[str] = []
    for doc in docs:
        name = doc.meta.get("nazwa")
        if name and name not in seen:
            seen.append(name)
    return seen


def evaluate(variant_name: str, searcher: Searcher, queries: list[Query], k: int) -> EvalResult:
    per_query = []
    for q in queries:
        ranked = _ranked_course_names(searcher.search(q.text))
        per_query.append(
            QueryResult(
                query=q.text,
                ranked=ranked,
                reciprocal_rank=reciprocal_rank(ranked, set(q.relevant)),
                ndcg=ndcg_at_k(ranked, q.relevant, k),
            )
        )
    return EvalResult(variant=variant_name, k=k, per_query=per_query)
