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
from search.eval.runner import EvalResult, evaluate
from search.eval.qrels import QUERIES
from search.eval.bench import BenchResult, bench
from search.eval.optimize import Candidate, optimize, GRIDS




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


def _print_eval_result(result: EvalResult) -> None:
    print(f"[{result.variant}] MRR={result.mrr:.3f}  NDCG@{result.k}={result.mean_ndcg:.3f}")
    for r in result.per_query:
        top = r.ranked[0] if r.ranked else "(brak wyników)"
        print(f"  RR={r.reciprocal_rank:.3f}  NDCG@{result.k}={r.ndcg:.3f}  {r.query!r}  -> {top!r}")



def cmd_eval(args: argparse.Namespace) -> None:
    if not args.all and not args.wariant:
        sys.exit("Podaj wariant albo użyj --all.")

    variants = list(VARIANTS.values()) if args.all else [get(args.wariant)]
    results = []
    for variant in variants:
        try:
            searcher = variant.searcher()
        except IndexNotReadyError as e:
            print(f"[{variant.name}] pominięto: {e}")
            continue
        except Exception as e:  # np. Qdrant/Chroma niedostępne — nie przerywaj oceny reszty wariantów
            if not args.all:
                raise
            print(f"[{variant.name}] pominięto: {e!r}")
            continue
        result = evaluate(variant.name, searcher, QUERIES, k=args.k)
        _print_eval_result(result)
        results.append(result)
        print()

    if args.all and len(results) > 1:
        print(f"Porównanie (na {len(QUERIES)} zapytaniach):")
        for result in sorted(results, key=lambda r: r.mrr, reverse=True):
            print(f"  {result.variant:15} MRR={result.mrr:.3f}  NDCG@{result.k}={result.mean_ndcg:.3f}")


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


def _print_bench_result(result: BenchResult) -> None:
    print(
        f"[{result.variant}] mediana={result.median_s * 1000:.0f}ms  p95={result.p95_s * 1000:.0f}ms  "
        f"min={result.min_s * 1000:.0f}ms  max={result.max_s * 1000:.0f}ms  "
        f"({result.n_queries} zapytań x {result.repeats} powtórzeń)"
    )


def cmd_bench(args: argparse.Namespace) -> None:
    if not args.all and not args.wariant:
        sys.exit("Podaj wariant albo użyj --all.")

    variants = list(VARIANTS.values()) if args.all else [get(args.wariant)]
    results = []
    for variant in variants:
        try:
            searcher = variant.searcher()
        except IndexNotReadyError as e:
            print(f"[{variant.name}] pominięto: {e}")
            continue
        except Exception as e:  # np. Qdrant/Chroma niedostępne — nie przerywaj benchmarku reszty wariantów
            if not args.all:
                raise
            print(f"[{variant.name}] pominięto: {e!r}")
            continue
        result = bench(variant.name, searcher, repeats=args.repeats)
        _print_bench_result(result)
        results.append(result)

    if args.all and len(results) > 1:
        print(f"\nPorównanie latencji (mediana, na {results[0].n_queries} zapytaniach):")
        for result in sorted(results, key=lambda r: r.median_s):
            print(f"  {result.variant:15} mediana={result.median_s * 1000:.0f}ms  p95={result.p95_s * 1000:.0f}ms")



def _print_candidates(variant_name: str, candidates: list[Candidate], k: int) -> None:
    print(f"[{variant_name}] parametry wg MRR (na {len(QUERIES)} zapytaniach):")
    for i, c in enumerate(candidates):
        marker = "  <- najlepszy" if i == 0 else ""
        print(f"  MRR={c.result.mrr:.3f}  NDCG@{k}={c.result.mean_ndcg:.3f}  {c.label}{marker}")


def cmd_optimize(args: argparse.Namespace) -> None:
    if not args.all and not args.wariant:
        sys.exit("Podaj wariant albo użyj --all.")

    names = list(GRIDS) if args.all else [args.wariant]
    for name in names:
        variant = get(name)
        try:
            candidates = optimize(variant, k=args.k)
        except IndexNotReadyError as e:
            print(f"[{variant.name}] pominięto: {e}")
            continue
        except Exception as e:  # np. Qdrant/Chroma niedostępne — nie przerywaj przeglądu reszty wariantów
            if not args.all:
                raise
            print(f"[{variant.name}] pominięto: {e!r}")
            continue
        _print_candidates(variant.name, candidates, args.k)
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

    p_eval = sub.add_parser("eval", help="Oceń trafność wariantu metrykami MRR/NDCG@k")
    p_eval.add_argument("wariant", nargs="?", choices=sorted(VARIANTS))
    p_eval.add_argument("--all", action="store_true", help="Oceń wszystkie warianty i porównaj wyniki")
    p_eval.add_argument("--k", type=int, default=5, help="Głębokość NDCG@k (domyślnie 5)")
    p_eval.set_defaults(func=cmd_eval)

    p_bench = sub.add_parser("bench", help="Zmierz latencję wyszukiwania wariantu")
    p_bench.add_argument("wariant", nargs="?", choices=sorted(VARIANTS))
    p_bench.add_argument("--all", action="store_true", help="Zmierz wszystkie warianty i porównaj wyniki")
    p_bench.add_argument("--repeats", type=int, default=3, help="Liczba powtórzeń całego zbioru zapytań (domyślnie 3)")
    p_bench.set_defaults(func=cmd_bench)

    p_optimize = sub.add_parser("optimize", help="Przeszukaj siatkę parametrów query-time wariantu wg MRR")
    p_optimize.add_argument("wariant", nargs="?", choices=sorted(GRIDS))
    p_optimize.add_argument("--all", action="store_true", help="Przeszukaj siatki wszystkich wariantów")
    p_optimize.add_argument("--k", type=int, default=5, help="Głębokość NDCG@k (domyślnie 5)")
    p_optimize.set_defaults(func=cmd_optimize)


    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
