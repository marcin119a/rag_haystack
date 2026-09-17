
from __future__ import annotations

from dataclasses import dataclass

from agno.agent import Agent
from agno.eval.accuracy import AccuracyEval, AccuracyResult

from agents.continuation.agent import continuation_agent
from agents.course.agent import catalog_agent
from agents.eval.cases import CASES, Case
from agents.model import build_model
from agents.triage.agent import triage_agent
from agents.workflow import db

AGENTS: dict[str, Agent] = {
    "triage": triage_agent,
    "catalog": catalog_agent,
    "continuation": continuation_agent,
}


@dataclass(frozen=True)
class CaseRun:
    case: Case
    result: AccuracyResult | None


def run_case(case: Case) -> CaseRun:
    evaluation = AccuracyEval(
        db=db,
        name=f"{case.agent_id}: {case.input[:60]}",
        model=build_model(),  # sędzia — ten sam model co agenci, dla spójności z resztą projektu
        agent=AGENTS[case.agent_id],
        input=case.input,
        expected_output=case.expected_output,
        additional_guidelines=case.guidelines,
        num_iterations=1,
    )
   
    result = evaluation.run(print_summary=False, print_results=False)
    return CaseRun(case=case, result=result)


def run_cases(cases: list[Case] | None = None, *, agent_id: str | None = None) -> list[CaseRun]:
    selected = [c for c in (cases or CASES) if agent_id is None or c.agent_id == agent_id]
    return [run_case(c) for c in selected]
