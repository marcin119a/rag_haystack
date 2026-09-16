from __future__ import annotations

from haystack import Document, Pipeline
from haystack.components.joiners import DocumentJoiner
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever

from search.base import IndexNotReadyError, Searcher
from search.variants.chroma_dense.indexer import COLLECTION, MODEL, connect_store


class ChromaDenseSearcher(Searcher):
    def __init__(
        self,
        *,
        top_k_bm25: int = 10,
        top_k_embedding: int = 10,
        top_k: int = 5,
        weights: list[float] | None = None,
    ) -> None:
        self._store = connect_store()
        if self._store.count_documents() == 0:
            raise IndexNotReadyError(
                f"Kolekcja {COLLECTION} jest pusta. Zbuduj indeks: uv run search index chroma_dense"
            )
        self._bm25_store = InMemoryDocumentStore()
        self._bm25_store.write_documents(self._store.filter_documents())
        self._top_k_bm25 = top_k_bm25
        self._top_k_embedding = top_k_embedding
        self._top_k = top_k
        self._weights = weights
        self._pipeline = self._create_pipeline()

    def _create_pipeline(self) -> Pipeline:
        text_embedder = SentenceTransformersTextEmbedder(model=MODEL)
        text_embedder.warm_up()

        pipeline = Pipeline()
        pipeline.add_component("text_embedder", text_embedder)
        pipeline.add_component(
            "embedding", ChromaEmbeddingRetriever(document_store=self._store, top_k=self._top_k_embedding)
        )
        pipeline.add_component("bm25", InMemoryBM25Retriever(self._bm25_store, top_k=self._top_k_bm25))
        pipeline.add_component(
            "joiner",
            DocumentJoiner(join_mode="reciprocal_rank_fusion", top_k=self._top_k, weights=self._weights),
        )
        pipeline.connect("text_embedder.embedding", "embedding.query_embedding")
        pipeline.connect("bm25", "joiner")
        pipeline.connect("embedding", "joiner")
        return pipeline

    def search(self, query: str) -> list[Document]:
        result = self._pipeline.run({"text_embedder": {"text": query}, "bm25": {"query": query}})
        return result["joiner"]["documents"]


def build_searcher(**kwargs) -> ChromaDenseSearcher:
    return ChromaDenseSearcher(**kwargs)
