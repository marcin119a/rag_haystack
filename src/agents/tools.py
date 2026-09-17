from __future__ import annotations

from functools import lru_cache

from haystack import Document

from search.base import IndexNotReadyError
from search.registry import get as get_variant


@lru_cache
def _catalog_searcher():
    return get_variant("openai_memory").searcher()


@lru_cache
def _program_searcher():
    return get_variant("qdrant_hybrid").searcher()


@lru_cache
def _related_searcher():
    return get_variant("neo4j_graph").searcher()


def _format_courses(docs: list[Document]) -> str:
    return "\n\n".join(
        f"{doc.content}\n"
        f"Kategoria: {doc.meta['kategoria']}, dni: {doc.meta['dni']}, PDF: {doc.meta['pdf_url']}"
        for doc in docs
    )


def search_catalog(query: str) -> str:
    """Wyszukuje szkolenia w katalogu (hybrydowo: BM25 + embeddingi OpenAI).

    Args:
        query: temat lub umiejętność, o którą pyta użytkownik.

    Returns:
        Pasujące szkolenia: nazwa z opisem, kategoria, liczba dni i link do PDF.
    """
    try:
        docs = _catalog_searcher().search(query)
    except IndexNotReadyError as e:
        return f"Wyszukiwarka katalogu nie jest gotowa: {e}"
    if not docs:
        return "Brak pasujących szkoleń w katalogu."
    return _format_courses(docs)


def search_program_fragments(query: str) -> str:
    """Wyszukuje fragmenty programów szkoleń (agendy, szczegółowa treść zajęć) w Qdrant.

    Args:
        query: temat lub zagadnienie, którego ma dotyczyć fragment programu.

    Returns:
        Pasujące fragmenty z nazwą pliku/szkolenia i treścią fragmentu.
    """
    try:
        docs = _program_searcher().search(query)
    except IndexNotReadyError as e:
        return f"Wyszukiwarka programów nie jest gotowa: {e}"
    if not docs:
        return "Brak pasujących fragmentów programów."
    return "\n\n".join(f"Plik: {doc.meta['plik']}\n{doc.content}" for doc in docs)


def search_related_trainings(query: str) -> str:
    """Wyszukuje szkolenia semantycznie powiązane z zapytaniem w indeksie Neo4j.

    Indeks jest zbudowany na tych samych danych co katalog, w grafie zawierającym też relacje
    PODOBNE_DO z sages.pl — dobry wybór, gdy user szuka alternatyw/odpowiedników
    konkretnego szkolenia, a nie dosłownego dopasowania słów kluczowych.

    Args:
        query: nazwa lub temat szkolenia, dla którego szukamy powiązanych/podobnych.

    Returns:
        Powiązane szkolenia: nazwa z opisem, kategoria, liczba dni i link do PDF.
    """
    try:
        docs = _related_searcher().search(query)
    except IndexNotReadyError as e:
        return f"Wyszukiwarka powiązanych szkoleń nie jest gotowa: {e}"
    if not docs:
        return "Brak powiązanych szkoleń."
    return _format_courses(docs)
