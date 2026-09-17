
from __future__ import annotations

from haystack import Document, Pipeline, component
from haystack.components.joiners import DocumentJoiner
from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersTextEmbedder
from neo4j_haystack import Neo4jDynamicDocumentRetriever, Neo4jEmbeddingRetriever

from search.base import IndexNotReadyError, Searcher
from search.variants.neo4j_semantic.indexer import FULLTEXT_INDEX, INDEX, MODEL, NODE_LABEL, RELATIONSHIP, connect_store


def _bm25_query(top_k: int) -> str:
    # db.index.fulltext.queryNodes zwraca (node, score) posortowane wg score (Lucene BM25Similarity).
    # top_k wypalony w tekst zapytania (stały na czas życia searchera) — $search_text jako parametr.
    return (
        f"CALL db.index.fulltext.queryNodes('{FULLTEXT_INDEX}', $search_text) YIELD node, score "
        f"RETURN node{{.*, score: score}} AS doc "
        f"LIMIT {top_k}"
    )


def _neighbors_query(top_k: int) -> str:
    return (
        f"MATCH (s:{NODE_LABEL} {{nazwa: $nazwa}})"
        f"-[:{RELATIONSHIP}]->(neighbor:{NODE_LABEL}) "
        f"WHERE neighbor.embedding IS NOT NULL "
        f"WITH collect(neighbor) AS neighbors "
        f"CALL db.index.fulltext.queryNodes("
        f"'{FULLTEXT_INDEX}', $query_text"
        f") YIELD node, score "
        f"WHERE node IN neighbors "
        f"WITH node, score AS bm25_score, "
        f"vector.similarity.cosine("
        f"node.embedding, $query_embedding"
        f") AS cosine_score "
        f"RETURN node{{"
        f".*, "
        f"bm25_score: bm25_score, "
        f"cosine_score: cosine_score"
        f"}} AS doc "
        f"ORDER BY bm25_score DESC "
        f"LIMIT {int(top_k)}"
    )


@component
class _AnchorNazwa:
    """Wyciąga meta['nazwa'] najlepiej dopasowanego dokumentu (rank 0) z gałęzi embedding — to ta
    kotwica, dla której neighbors odpytuje graf o krawędzie :PODOBNE_DO."""

    @component.output_types(nazwa=str)
    def run(self, documents: list[Document]) -> dict[str, str]:
        return {"nazwa": documents[0].meta["nazwa"] if documents else ""}


class Neo4jNeighborsSearcher(Searcher):
    def __init__(
        self,
        *,
        top_k_bm25: int = 10,
        top_k_embedding: int = 10,
        top_k_neighbors: int = 10,
        top_k: int = 10,
        weights: list[float] | None = None,
    ) -> None:
        # weights: [waga bm25, waga embedding, waga sąsiadów z grafu] dla fuzji RRF; None = równe
        # wagi (domyślnie). Parametry query-time (nie wymagają przebudowy indeksu) — patrz
        # search.eval.optimize.
        self._store = connect_store()
        if self._store.count_documents() == 0:
            raise IndexNotReadyError(f"Indeks {INDEX} jest pusty. Zbuduj go: uv run search index neo4j_neighbors")
        self._top_k_bm25 = top_k_bm25
        self._top_k_embedding = top_k_embedding
        self._top_k_neighbors = top_k_neighbors
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
        pipeline.add_component("anchor", _AnchorNazwa())
        pipeline.add_component(
            "neighbors",
            Neo4jDynamicDocumentRetriever(
                client_config=self._store.client_config,
                query=_neighbors_query(self._top_k_neighbors),
                runtime_parameters=["nazwa", "query_embedding", "query_text"],
            ),
        )
        pipeline.add_component(
            "joiner",
            DocumentJoiner(join_mode="reciprocal_rank_fusion", top_k=self._top_k, weights=self._weights),
        )
        pipeline.connect("text_embedder.embedding", "embedding.query_embedding")
        pipeline.connect("text_embedder.embedding", "neighbors.query_embedding")
        pipeline.connect("embedding.documents", "anchor.documents")
        pipeline.connect("anchor.nazwa", "neighbors.nazwa")
        pipeline.connect("bm25", "joiner")
        pipeline.connect("embedding", "joiner")
        pipeline.connect("neighbors", "joiner")
        return pipeline

    def search(self, query: str) -> list[Document]:
        result = self._pipeline.run(
            {"text_embedder": {"text": query}, "bm25": {"search_text": query}, "neighbors": {"query_text": query}}
        )
        return result["joiner"]["documents"]


def build_searcher(**kwargs) -> Neo4jNeighborsSearcher:
    return Neo4jNeighborsSearcher(**kwargs)
