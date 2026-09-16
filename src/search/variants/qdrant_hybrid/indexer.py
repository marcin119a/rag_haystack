
from __future__ import annotations

import sys

from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.embedders.fastembed import FastembedSparseDocumentEmbedder
from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersDocumentEmbedder
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from sentence_transformers import SentenceTransformer

from search.base import Indexer
from search.loaders import load_program_docs, split_program_sections
from settings import settings

MODEL = settings.embedding_model
SPARSE_MODEL = "Qdrant/bm25"
SPARSE_KWARGS = {"disable_stemmer": True}
COLLECTION = "szkolenia_chunki"


def connect_store() -> QdrantDocumentStore:
    # Wymiar wektora gęstego zależy od modelu — liczymy go raz, zamiast wpisywać na sztywno.
    embedding_dim = SentenceTransformer(MODEL).get_sentence_embedding_dimension()
    return QdrantDocumentStore(
        url=settings.qdrant_url,
        index=COLLECTION,
        embedding_dim=embedding_dim,
        use_sparse_embeddings=True,
        sparse_idf=True,  # IDF dla BM25 liczy Qdrant po swojej stronie
    )


class QdrantHybridIndexer(Indexer):
    def __init__(self) -> None:
        self.store = connect_store()

    def is_ready(self) -> bool:
        return self.store.count_documents() > 0

    def build(self) -> None:
        docs = load_program_docs()
        if not docs:
            sys.exit(f"Brak plików .md w {settings.programy_dir} (rozpakuj data_rag/programy.zip).")

        chunks = split_program_sections(docs)
        print(f"{len(docs)} plików -> {len(chunks)} fragmentów", flush=True)

        if self.store.count_documents() == len({chunk.id for chunk in chunks}):
            print(f"Kolekcja {COLLECTION} jest już zaindeksowana.", flush=True)
            return

        doc_embedder = SentenceTransformersDocumentEmbedder(model=MODEL)
        doc_embedder.warm_up()
        sparse_doc_embedder = FastembedSparseDocumentEmbedder(model=SPARSE_MODEL, model_kwargs=SPARSE_KWARGS)
        sparse_doc_embedder.warm_up()

        dense_documents = doc_embedder.run(chunks)["documents"]
        embedded = sparse_doc_embedder.run(dense_documents)["documents"]

        self.store.write_documents(embedded, policy=DuplicatePolicy.OVERWRITE)
        print(f"Zapisano {self.store.count_documents()} fragmentów do kolekcji {COLLECTION}.", flush=True)


def build_indexer() -> QdrantHybridIndexer:
    return QdrantHybridIndexer()
