from __future__ import annotations

from abc import ABC, abstractmethod

from haystack import Document


class IndexNotReadyError(RuntimeError):
    """Indeks tego wariantu jeszcze nie istnieje albo jest pusty.

    Nie jest to błąd wyszukiwania — trzeba najpierw zbudować indeks, np.:
        uv run search index <wariant>
    """


class Indexer(ABC):
    """Buduje indeks dla jednego wariantu. Wywoływany jawnie (CLI, usługa indexer) — nigdy przez Searcher."""

    @abstractmethod
    def is_ready(self) -> bool:
        """Czy indeks już istnieje i można po nim wyszukiwać."""

    @abstractmethod
    def build(self) -> None:
        """(Prz)buduj indeks od zera."""


class Searcher(ABC):


    @abstractmethod
    def search(self, query: str) -> list[Document]: ...