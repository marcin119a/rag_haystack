
from __future__ import annotations

import argparse
import sys

from agents.eval.cases import CASES
from agents.eval.runner import AGENTS, run_cases


def cmd_list(_args: argparse.Namespace) -> None:
    for case in CASES:
        print(f"{case.agent_id:12} {case.input}")


def cmd_run(args: argparse.Namespace) -> None:
    if not args.all and not args.agent:
        sys.exit("Podaj agenta (triage|catalog|continuation) albo użyj --all.")

    runs = run_cases(agent_id=None if args.all else args.agent)
    if not runs:
        sys.exit("Brak przypadków do uruchomienia.")

    for run in runs:
        result = run.result
        if result is None or not result.results:
            print(f"[{run.case.agent_id}] BŁĄD: ewaluacja nie zwróciła wyniku dla {run.case.input!r}")
            continue
        print(f"[{run.case.agent_id}] avg_score={result.avg_score:.1f}/10  {run.case.input!r}")
        for entry in result.results:
            print(f"    output: {str(entry.output).replace(chr(10), ' ')}")
            print(f"    reason: {entry.reason}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agents-eval")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="Wypisz zdefiniowane przypadki").set_defaults(func=cmd_list)

    run_parser = sub.add_parser("run", help="Uruchom AccuracyEval (LLM-as-judge)")
    run_parser.add_argument("agent", nargs="?", choices=list(AGENTS), help="triage | catalog | continuation")
    run_parser.add_argument("--all", action="store_true", help="Uruchom dla wszystkich agentów")
    run_parser.set_defaults(func=cmd_run)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
