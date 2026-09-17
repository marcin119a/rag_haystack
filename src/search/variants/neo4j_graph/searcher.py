
from __future__ import annotations

from haystack import Document, Pipeline, component
from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersTextEmbedder
from neo4j_haystack import Neo4jDynamicDocumentRetriever, Neo4jEmbeddingRetriever

from search.base import IndexNotReadyError, Searcher
from search.variants.neo4j_semantic.indexer import INDEX, MODEL, NODE_LABEL, RELATIONSHIP, connect_store


def _neighbors_query(top_k: int) -> str:
    return (
        f"MATCH (s:{NODE_LABEL} {{nazwa: $nazwa}})-[:{RELATIONSHIP}]->(t:{NODE_LABEL}) "
        f"WITH t, vector.similarity.cosine(t.embedding, $query_embedding) AS score "
        f"RETURN t{{.*, score: score}} AS doc "
        f"ORDER BY score DESC "
        f"LIMIT {top_k}"
    )


@component
class _AnchorNazwa:

    @component.output_types(nazwa=str)
    def run(self, documents: list[Document]) -> dict[str, str]:
        return {"nazwa": documents[0].meta["nazwa"] if documents else ""}


class Neo4jGraphSearcher(Searcher):
    def __init__(self, *, top_k: int = 10) -> None:
        # Jedyny parametr query-time: ilu sąsiadów z grafu zwrócić (patrz search.eval.optimize).
        self._store = connect_store()
        if self._store.count_documents() == 0:
            raise IndexNotReadyError(f"Indeks {INDEX} jest pusty. Zbuduj go: uv run search index neo4j_graph")
        self._top_k = top_k
        self._pipeline = self._create_pipeline()

    def _create_pipeline(self) -> Pipeline:
        text_embedder = SentenceTransformersTextEmbedder(model=MODEL)
        text_embedder.warm_up()

        pipeline = Pipeline()
        pipeline.add_component("text_embedder", text_embedder)
        pipeline.add_component("embedding", Neo4jEmbeddingRetriever(document_store=self._store, top_k=1))
        pipeline.add_component("anchor", _AnchorNazwa())
        pipeline.add_component(
            "neighbors",
            Neo4jDynamicDocumentRetriever(
                client_config=self._store.client_config,
                query=_neighbors_query(self._top_k),
                runtime_parameters=["nazwa", "query_embedding"],
            ),
        )
        pipeline.connect("text_embedder.embedding", "embedding.query_embedding")
        pipeline.connect("text_embedder.embedding", "neighbors.query_embedding")
        pipeline.connect("embedding.documents", "anchor.documents")
        pipeline.connect("anchor.nazwa", "neighbors.nazwa")
        return pipeline

    def search(self, query: str) -> list[Document]:
        result = self._pipeline.run({"text_embedder": {"text": query}})
        return result["neighbors"]["documents"]


def build_searcher(**kwargs) -> Neo4jGraphSearcher:
    return Neo4jGraphSearcher(**kwargs)
