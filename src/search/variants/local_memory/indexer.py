from __future__ import annotations

from pathlib import Path

from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersDocumentEmbedder

from search.base import Indexer
from search.loaders import load_course_docs
from settings import settings

MODEL = settings.embedding_model
INDEX_PATH = Path(settings.index_path)


class LocalMemoryIndexer(Indexer):
    def __init__(self, path: Path = INDEX_PATH) -> None:
        self.path = path

    def is_ready(self) -> bool:
        return self.path.exists()

    def build(self) -> None:
        store = InMemoryDocumentStore()
        doc_embedder = SentenceTransformersDocumentEmbedder(model=MODEL)
        doc_embedder.warm_up()
        store.write_documents(doc_embedder.run(load_course_docs())["documents"], policy=DuplicatePolicy.OVERWRITE)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        store.save_to_disk(str(self.path))


def build_indexer() -> LocalMemoryIndexer:
    return LocalMemoryIndexer()
