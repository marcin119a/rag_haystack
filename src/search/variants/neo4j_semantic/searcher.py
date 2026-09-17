from __future__ import annotations

from haystack import Document, Pipeline
from haystack.components.joiners import DocumentJoiner
from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersTextEmbedder
from neo4j_haystack import Neo4jDynamicDocumentRetriever, Neo4jEmbeddingRetriever

from search.base import IndexNotReadyError, Searcher
from search.variants.neo4j_semantic.indexer import FULLTEXT_INDEX, INDEX, MODEL, connect_store


def _bm25_query(top_k: int) -> str:
    return (
        f"CALL db.index.fulltext.queryNodes('{FULLTEXT_INDEX}', $search_text) YIELD node, score "
        f"RETURN node{{.*, score: score}} AS doc "
        f"LIMIT {top_k}"
    )


class Neo4jSemanticSearcher(Searcher):
    def __init__(
        self,
        *,
        top_k_bm25: int = 10,
        top_k_embedding: int = 10,
        top_k: int = 10,
        weights: list[float] | None = None,
    ) -> None:
        # weights: [waga bm25, waga embedding] dla fuzji RRF; None = równe wagi (domyślnie).
        # Parametry query-time (nie wymagają przebudowy indeksu) — patrz search.eval.optimize.
        self._store = connect_store()
        if self._store.count_documents() == 0:
            raise IndexNotReadyError(f"Indeks {INDEX} jest pusty. Zbuduj go: uv run search index neo4j_semantic")
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
            "embedding", Neo4jEmbeddingRetriever(document_store=self._store, top_k=self._top_k_embedding)
        )
        pipeline.add_component(
            "bm25",
            Neo4jDynamicDocumentRetriever(
                client_config=self._store.client_config,
                query=_bm25_query(self._top_k_bm25),
                runtime_parameters=["search_text"],
            ),
        )
        pipeline.add_component(
            "joiner",
            DocumentJoiner(join_mode="reciprocal_rank_fusion", top_k=self._top_k, weights=self._weights),
        )
        pipeline.connect("text_embedder.embedding", "embedding.query_embedding")
        pipeline.connect("bm25", "joiner")
        pipeline.connect("embedding", "joiner")
        return pipeline

    def search(self, query: str) -> list[Document]:
        result = self._pipeline.run({"text_embedder": {"text": query}, "bm25": {"search_text": query}})
        return result["joiner"]["documents"]


def build_searcher(**kwargs) -> Neo4jSemanticSearcher:
    return Neo4jSemanticSearcher(**kwargs)