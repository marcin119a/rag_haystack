from __future__ import annotations

from haystack import Document, Pipeline
from haystack.components.embedders import OpenAITextEmbedder
from haystack_integrations.components.embedders.fastembed import FastembedSparseTextEmbedder
from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.qdrant import QdrantHybridRetriever

from search.base import IndexNotReadyError, Searcher
from search.variants.qdrant_hybrid.indexer import COLLECTION, MODEL, SPARSE_KWARGS, SPARSE_MODEL, connect_store, API_KEY
from settings import settings


class QdrantHybridSearcher(Searcher):
    def __init__(self, *, top_k: int = 10, rrf_weights: list[float] = (0.4, 0.6)) -> None:
        # Parametry query-time (nie wymagają przebudowy indeksu) — patrz search.eval.optimize.
        self._store = connect_store()
        if self._store.count_documents() == 0:
            raise IndexNotReadyError(
                f"Kolekcja {COLLECTION} jest pusta. Zbuduj indeks: uv run search index qdrant_hybrid"
            )
        self._top_k = top_k
        self._rrf_weights = list(rrf_weights)
        self._pipeline = self._create_pipeline()

    def _create_pipeline(self) -> Pipeline:
        # Te same modele co w indexer.py — zapytanie musi być zembedowane tak samo jak fragmenty.
        text_embedder = OpenAITextEmbedder(api_key=API_KEY, model=MODEL)
        text_embedder.warm_up()
        sparse_text_embedder = FastembedSparseTextEmbedder(model=SPARSE_MODEL, model_kwargs=SPARSE_KWARGS)
        sparse_text_embedder.warm_up()

        pipeline = Pipeline()
        pipeline.add_component("text_embedder", text_embedder)
        pipeline.add_component("sparse_embedder", sparse_text_embedder)
        pipeline.add_component(
            "retriever",
            QdrantHybridRetriever(document_store=self._store, top_k=self._top_k, rrf_weights=self._rrf_weights),
        )
        pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        pipeline.connect("sparse_embedder.sparse_embedding", "retriever.query_sparse_embedding")
        return pipeline

    def search(self, query: str) -> list[Document]:
        result = self._pipeline.run({"text_embedder": {"text": query}, "sparse_embedder": {"text": query}})
        return result["retriever"]["documents"]


def build_searcher(**kwargs) -> QdrantHybridSearcher:
    return QdrantHybridSearcher(**kwargs)