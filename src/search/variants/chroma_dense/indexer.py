from __future__ import annotations

import sys

from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersDocumentEmbedder
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

from search.base import Indexer
from search.loaders import load_program_docs, recursive_split
from settings import settings

MODEL = settings.embedding_model
COLLECTION = "szkolenia_chunki"


def connect_store() -> ChromaDocumentStore:
    return ChromaDocumentStore(
        collection_name=COLLECTION,
        persist_path=settings.chroma_path,
        distance_function="cosine",
    )


class ChromaDenseIndexer(Indexer):
    def __init__(self) -> None:
        self.store = connect_store()

    def is_ready(self) -> bool:
        return self.store.count_documents() > 0

    def build(self) -> None:
        docs = load_program_docs()
        if not docs:
            sys.exit(f"Brak plików .md w {settings.programy_dir} (rozpakuj data_rag/programy.zip).")

        chunks = recursive_split(docs)
        for chunk in chunks:
            # Chroma przyjmuje w metadanych tylko str/int/float/bool (i ich listy); splitter dokłada listę
            # słowników, której i tak nie używamy — usuwamy, żeby nie dostać ostrzeżenia przy każdym fragmencie.
            chunk.meta.pop("_split_overlap", None)

        print(f"{len(docs)} plików -> {len(chunks)} fragmentów", flush=True)

        if self.store.count_documents() == len({chunk.id for chunk in chunks}):
            print(f"Kolekcja {COLLECTION} w {settings.chroma_path} jest już zaindeksowana.", flush=True)
            return

        doc_embedder = SentenceTransformersDocumentEmbedder(model=MODEL)
        doc_embedder.warm_up()
        embedded = doc_embedder.run(chunks)["documents"]

        self.store.write_documents(embedded, policy=DuplicatePolicy.OVERWRITE)
        print(
            f"Zapisano {self.store.count_documents()} fragmentów do kolekcji {COLLECTION} w {settings.chroma_path}.",
            flush=True,
        )


def build_indexer() -> ChromaDenseIndexer:
    return ChromaDenseIndexer()