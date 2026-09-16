from __future__ import annotations

from haystack import Document, Pipeline
from haystack.components.embedders import OpenAITextEmbedder
from haystack.components.joiners import DocumentJoiner
from haystack.components.retrievers.in_memory import (
    InMemoryBM25Retriever,
    InMemoryEmbeddingRetriever,
)
from haystack.document_stores.in_memory import InMemoryDocumentStore

from search.base import IndexNotReadyError, Searcher
from search.variants.openai_memory.indexer import API_KEY, INDEX_PATH, MODEL


class OpenAiMemorySearcher(Searcher):
    def __init__(
        self,
        *,
        top_k_bm25: int = 10,
        top_k_embedding: int = 10,
        top_k: int = 5,
        weights: list[float] | None = None,
    ) -> None:
        if not INDEX_PATH.exists():
            raise IndexNotReadyError(
                f"Brak indeksu {INDEX_PATH}. Zbuduj go: uv run search index openai_memory"
            )
        self._store = InMemoryDocumentStore.load_from_disk(str(INDEX_PATH))
        self._top_k_bm25 = top_k_bm25
        self._top_k_embedding = top_k_embedding
        self._top_k = top_k
        self._weights = weights   
        self._pipeline = self._create_pipeline()

    def _create_pipeline(self) -> Pipeline:
        pipeline = Pipeline()
        pipeline.add_component("text_embedder", OpenAITextEmbedder(api_key=API_KEY, model=MODEL))
        pipeline.add_component("bm25", InMemoryBM25Retriever(self._store, top_k=self._top_k_bm25))
        pipeline.add_component("embedding", InMemoryEmbeddingRetriever(self._store, top_k=self._top_k_embedding))
        pipeline.add_component("joiner", DocumentJoiner(join_mode="reciprocal_rank_fusion", top_k=self._top_k))
        pipeline.connect("text_embedder.embedding", "embedding.query_embedding")
        pipeline.connect("bm25", "joiner")
        pipeline.connect("embedding", "joiner")
        return pipeline

    def search(self, query: str) -> list[Document]:
        result = self._pipeline.run({"text_embedder": {"text": query}, "bm25": {"query": query}})
        return result["joiner"]["documents"]


def build_searcher(**kwargs) -> OpenAiMemorySearcher:
    return OpenAiMemorySearcher(**kwargs)
