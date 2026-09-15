"""CLI do wariantów wyszukiwania — po jednej komendzie na czynność, indeksowanie osobno od wyszukiwania:

    uv run search list
    uv run search index <wariant> [--force]
    uv run search search <wariant> [zapytanie...]
"""

from __future__ import annotations

import argparse
import sys

from search.base import IndexNotReadyError
from search.registry import VARIANTS, get


def cmd_list(_args: argparse.Namespace) -> None:
    for variant in VARIANTS.values():
        print(f"{variant.name:15} {variant.description}")


def cmd_index(args: argparse.Namespace) -> None:
    variant = get(args.wariant)
    indexer = variant.indexer()
    if indexer.is_ready() and not args.force:
        print(f"Indeks wariantu {variant.name!r} już istnieje (użyj --force, żeby przebudować).")
        return
    indexer.build()


def cmd_search(args: argparse.Namespace) -> None:
    variant = get(args.wariant)
    try:
        searcher = variant.searcher()
    except IndexNotReadyError as e:
        sys.exit(str(e))

    query = " ".join(args.zapytanie) or "szkolenie z Docker"
    print(f"[{variant.name}] Zapytanie: {query}\n")
    for doc in searcher.search(query):
        title = doc.meta.get("nazwa", "")
        print(f"{doc.score:.4f}  {title}  ({doc.meta.get('kategoria')}, {doc.meta.get('dni')} dni)")
        print(f"        {doc.meta.get('pdf_url')}")
        fragment = doc.content.removeprefix(f"{title}\n\n") if title else doc.content
        if fragment and fragment != title:
            print("        " + fragment.replace("\n", "\n        "))
        print()


def main() -> None:
    parser = argparse.ArgumentParser(prog="search", description="Warianty wyszukiwania: indeksowanie i zapytania.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="Wypisz dostępne warianty")
    p_list.set_defaults(func=cmd_list)

    p_index = sub.add_parser("index", help="Zbuduj indeks wariantu")
    p_index.add_argument("wariant", choices=sorted(VARIANTS))
    p_index.add_argument("--force", action="store_true", help="Przebuduj, nawet jeśli indeks już istnieje")
    p_index.set_defaults(func=cmd_index)

    p_search = sub.add_parser("search", help="Wyszukaj w wariancie")
    p_search.add_argument("wariant", choices=sorted(VARIANTS))
    p_search.add_argument("zapytanie", nargs="*")
    p_search.set_defaults(func=cmd_search)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
