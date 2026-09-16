
from __future__ import annotations

from haystack import Document, Pipeline
from haystack.components.joiners import DocumentJoiner
from haystack.components.retrievers.in_memory import (
    InMemoryBM25Retriever,
    InMemoryEmbeddingRetriever,
)
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersTextEmbedder

from search.base import IndexNotReadyError, Searcher
from search.variants.local_memory.indexer import INDEX_PATH, MODEL


class LocalMemorySearcher(Searcher):
    def __init__(self, *args, **kwargs) -> None:
        if not INDEX_PATH.exists():
            raise IndexNotReadyError(
                f"Brak indeksu {INDEX_PATH}. Zbuduj go: uv run search index local_memory"
            )
        self._store = InMemoryDocumentStore.load_from_disk(str(INDEX_PATH))
        self._pipeline = self._create_pipeline()

    def _create_pipeline(self) -> Pipeline:
        pipeline = Pipeline()
        pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder(model=MODEL))
        pipeline.add_component("bm25", InMemoryBM25Retriever(self._store, top_k=10))
        pipeline.add_component("embedding", InMemoryEmbeddingRetriever(self._store, top_k=10))
        pipeline.add_component("joiner", DocumentJoiner(join_mode="reciprocal_rank_fusion", top_k=5))
        pipeline.connect("text_embedder.embedding", "embedding.query_embedding")
        pipeline.connect("bm25", "joiner")
        pipeline.connect("embedding", "joiner")
        return pipeline

    def search(self, query: str) -> list[Document]:
        result = self._pipeline.run({"text_embedder": {"text": query}, "bm25": {"query": query}})
        return result["joiner"]["documents"]


def build_searcher(**kwargs) -> LocalMemorySearcher:
    return LocalMemorySearcher(**kwargs)
