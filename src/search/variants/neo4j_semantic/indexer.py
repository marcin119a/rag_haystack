
from __future__ import annotations

import sys

import pandas as pd
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersDocumentEmbedder
from neo4j_haystack import Neo4jClientConfig, Neo4jDocumentStore, Neo4jQueryWriter
from sentence_transformers import SentenceTransformer

from search.base import Indexer
from search.loaders import load_course_docs
from settings import settings

MODEL = settings.embedding_model
NODE_LABEL = "Szkolenie"
INDEX = "szkolenia_opisy"
FULLTEXT_INDEX = "szkolenia_bm25"
RELATIONSHIP = "PODOBNE_DO"

_MERGE_EDGES_QUERY = f"""
UNWIND $rows AS row
MATCH (s:{NODE_LABEL} {{nazwa: row.source_nazwa}})
MATCH (t:{NODE_LABEL} {{nazwa: row.target_nazwa}})
MERGE (s)-[:{RELATIONSHIP}]->(t)
"""

_CREATE_FULLTEXT_INDEX_QUERY = f"""
CREATE FULLTEXT INDEX {FULLTEXT_INDEX} IF NOT EXISTS
FOR (n:{NODE_LABEL}) ON EACH [n.content]
"""


def _client_config() -> Neo4jClientConfig:
    return Neo4jClientConfig(
        url=settings.neo4j_url,
        database=settings.neo4j_database,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
    )


def connect_store() -> Neo4jDocumentStore:
    embedding_dim = SentenceTransformer(MODEL).get_sentence_embedding_dimension()
    return Neo4jDocumentStore(
        client_config=_client_config(),
        index=INDEX,
        node_label=NODE_LABEL,
        embedding_dim=embedding_dim,
        similarity="cosine",
    )


def _matched_edges(catalog_nazwy: set[str]) -> list[dict]:
    try:
        edges = pd.read_parquet(settings.podobne_szkolenia_parquet)
    except FileNotFoundError:
        sys.exit(f"Brak {settings.podobne_szkolenia_parquet}. Zbuduj go najpierw: uv run scrape-podobne")

    slug_to_nazwa = edges.drop_duplicates("target_slug").set_index("target_slug")["target_nazwa"].to_dict()

    rows = pd.DataFrame(
        {
            "source_nazwa": edges["source_slug"].map(slug_to_nazwa),
            "target_nazwa": edges["target_nazwa"],
        }
    ).dropna()
    rows = rows[rows["source_nazwa"].isin(catalog_nazwy) & rows["target_nazwa"].isin(catalog_nazwy)]
    return rows.drop_duplicates().to_dict("records")


class Neo4jSemanticIndexer(Indexer):
    def __init__(self) -> None:
        self.store = connect_store()

    def is_ready(self) -> bool:
        return self.store.count_documents() > 0

    def build(self) -> None:
        docs = load_course_docs()
        print(f"{len(docs)} szkoleń z katalogu -> węzły {NODE_LABEL}", flush=True)

        if self.store.count_documents() == len(docs):
            print(f"Węzły {NODE_LABEL} są już zaindeksowane (embeddingi pomijam).", flush=True)
        else:
            doc_embedder = SentenceTransformersDocumentEmbedder(model=MODEL)
            doc_embedder.warm_up()
            embedded = doc_embedder.run(docs)["documents"]
            self.store.write_documents(embedded, policy=DuplicatePolicy.OVERWRITE)
            print(f"Zapisano {self.store.count_documents()} węzłów {NODE_LABEL}.", flush=True)

        Neo4jQueryWriter(client_config=_client_config(), query=_CREATE_FULLTEXT_INDEX_QUERY).run(parameters={})
        print(f"Indeks pełnotekstowy {FULLTEXT_INDEX} gotowy (BM25 przez Lucene).", flush=True)

        catalog_nazwy = {doc.meta["nazwa"] for doc in docs}
        edge_rows = _matched_edges(catalog_nazwy)
        Neo4jQueryWriter(client_config=_client_config(), query=_MERGE_EDGES_QUERY).run(parameters={"rows": edge_rows})
        print(f"Zapisano {len(edge_rows)} relacji :{RELATIONSHIP} (dopasowanych do katalogu).", flush=True)


def build_indexer() -> Neo4jSemanticIndexer:
    return Neo4jSemanticIndexer()
