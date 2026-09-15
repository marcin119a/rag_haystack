
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module

from search.base import Indexer, Searcher


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    module: str 

    def indexer(self) -> Indexer:
        return import_module(f"{self.module}.indexer").build_indexer()

    def searcher(self) -> Searcher:
        return import_module(f"{self.module}.searcher").build_searcher()


_VARIANTS = [
    Variant(
        "local_memory",
        "Sentence-transformers + BM25, in-memory, pełne opisy szkoleń",
        "search.variants.local_memory",
    ),
    Variant(
        "openai_memory",
        "OpenAI embeddings + in-memory, pełne opisy szkoleń",
        "search.variants.openai_memory",
    ),
    Variant(
        "chroma_dense",
        "Sentence-transformers + Chroma, pełne opisy szkoleń",
        "search.variants.chroma_dense",
    ),
]

VARIANTS: dict[str, Variant] = {v.name: v for v in _VARIANTS}


def get(name: str) -> Variant:
    try:
        return VARIANTS[name]
    except KeyError:
        dostepne = ", ".join(VARIANTS)
        raise SystemExit(f"Nieznany wariant: {name!r}. Dostępne: {dostepne}") from None
