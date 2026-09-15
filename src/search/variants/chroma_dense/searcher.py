
from __future__ import annotations

from haystack import Document, Pipeline
from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.joiners import DocumentJoiner
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever

from search.base import IndexNotReadyError, Searcher
from search.variants.chroma_dense.indexer import COLLECTION, MODEL, connect_store


class ChromaDenseSearcher(Searcher):
    def __init__(self) -> None:
        self._store = connect_store()
        if self._store.count_documents() == 0:
            raise IndexNotReadyError(
                f"Kolekcja {COLLECTION} jest pusta. Zbuduj indeks: uv run search index chroma_dense"
            )
        self._bm25_store = InMemoryDocumentStore()
        self._bm25_store.write_documents(self._store.filter_documents())
        self._pipeline = self._create_pipeline()


    def _create_pipeline(self) -> Pipeline:
        text_embedder = SentenceTransformersTextEmbedder(model=MODEL)
        text_embedder.warm_up()

        pipeline = Pipeline()
        pipeline.add_component("text_embedder", text_embedder)
        pipeline.add_component("embedding", ChromaEmbeddingRetriever(document_store=self._store, top_k=10))
        pipeline.add_component("bm25", InMemoryBM25Retriever(self._bm25_store, top_k=10))
        pipeline.add_component("joiner", DocumentJoiner(join_mode="reciprocal_rank_fusion", top_k=5))
        pipeline.connect("text_embedder.embedding", "embedding.query_embedding")
        pipeline.connect("embedding", "joiner")
        pipeline.connect("bm25", "joiner")
        return pipeline

    def search(self, query: str) -> list[Document]:
        result = self._pipeline.run({"text_embedder": {"text": query}, "bm25": {"query": query}})
        return result["joiner"]["documents"]


def build_searcher() -> ChromaDenseSearcher:
    return ChromaDenseSearcher()
