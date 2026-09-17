from __future__ import annotations

import sys
from pathlib import Path

from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersDocumentEmbedder
from haystack_integrations.document_stores.faiss import FAISSDocumentStore

from search.base import Indexer
from search.loaders import load_program_docs, recursive_split
from settings import settings

MODEL = settings.embedding_model
EMBEDDING_DIM = 384  


def connect_store() -> FAISSDocumentStore:
    return FAISSDocumentStore(index_path=settings.faiss_path, embedding_dim=EMBEDDING_DIM)


class FaissDenseIndexer(Indexer):
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
            chunk.meta.pop("_split_overlap", None)

        print(f"{len(docs)} plików -> {len(chunks)} fragmentów", flush=True)

        doc_embedder = SentenceTransformersDocumentEmbedder(model=MODEL)
        doc_embedder.warm_up()
        embedded = doc_embedder.run(chunks)["documents"]

        self.store.write_documents(embedded, policy=DuplicatePolicy.OVERWRITE)
        Path(settings.faiss_path).parent.mkdir(parents=True, exist_ok=True)
        self.store.save(settings.faiss_path)
        print(
            f"Zapisano {self.store.count_documents()} fragmentów do indeksu FAISS w {settings.faiss_path}.",
            flush=True,
        )


def build_indexer() -> FaissDenseIndexer:
    return FaissDenseIndexer()
