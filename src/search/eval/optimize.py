from __future__ import annotations

from dataclasses import dataclass

from search.eval.qrels import QUERIES
from search.eval.runner import EvalResult, evaluate
from search.registry import Variant



GRIDS: dict[str, list[tuple[str, dict]]] = {
    "local_memory": [
        ("równe wagi (domyślne)", {}),
        ("bm25 60% / dense 40%", {"weights": [0.6, 0.4]}),
        ("bm25 40% / dense 60%", {"weights": [0.4, 0.6]}),
        ("top_k=3 po fuzji", {"top_k": 3}),
        ("top_k=10 po fuzji", {"top_k": 10}),
    ],
    "openai_memory": [
        ("równe wagi (domyślne)", {}),
        ("bm25 60% / dense 40%", {"weights": [0.6, 0.4]}),
        ("bm25 40% / dense 60%", {"weights": [0.4, 0.6]}),
        ("top_k=3 po fuzji", {"top_k": 3}),
    ],
    "chroma_dense": [
        ("równe wagi (domyślne)", {}),
        ("bm25 60% / dense 40%", {"weights": [0.6, 0.4]}),
        ("bm25 40% / dense 60%", {"weights": [0.4, 0.6]}),
        ("bm25 20% / dense 80%", {"weights": [0.2, 0.8]}),
        ("top_k=3 po fuzji", {"top_k": 3}),
        ("top_k=10 po fuzji", {"top_k": 10}),
    ],
    "qdrant_pdf": [
        ("top_k=5 (domyślne)", {}),
        ("top_k=3", {"top_k": 3}),
        ("top_k=10", {"top_k": 10}),
        ("rrf sparse 20% / dense 80%", {"rrf_weights": [0.2, 0.8]}),
    ],
    "qdrant_hybrid": [
        ("rrf sparse 40% / dense 60% (domyślne)", {}),
        ("rrf sparse 60% / dense 40%", {"rrf_weights": [0.6, 0.4]}),
        ("rrf sparse 20% / dense 80%", {"rrf_weights": [0.2, 0.8], "top_k": 10}),
        ("rrf sparse 20% / dense 80%", {"rrf_weights": [0.2, 0.8], "top_k": 5}),
        ("rrf równe wagi", {"rrf_weights": [0.5, 0.5]}),
        ("top_k=15", {"top_k": 15}),
    ],
}


@dataclass(frozen=True)
class Candidate:
    label: str
    kwargs: dict
    result: EvalResult


def optimize(variant: Variant, k: int = 5) -> list[Candidate]:
    """Ocenia siatkę GRIDS[variant.name] i zwraca kandydatów posortowanych malejąco wg MRR.

    Buduje searcher osobno dla każdego kandydata (ładuje ponownie model/łączy się z indeksem) —
    prostota ważniejsza niż szybkość siatki złożonej z kilku pozycji."""
    try:
        grid = GRIDS[variant.name]
    except KeyError:
        raise SystemExit(
            f"Brak siatki parametrów dla wariantu {variant.name!r}. Dostępne: {', '.join(GRIDS)}"
        ) from None

    candidates = []
    for label, kwargs in grid:
        searcher = variant.searcher(**kwargs)
        result = evaluate(variant.name, searcher, QUERIES, k=k)
        candidates.append(Candidate(label=label, kwargs=kwargs, result=result))
    return sorted(candidates, key=lambda c: c.result.mrr, reverse=True)
