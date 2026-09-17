
from agno.db.sqlite import SqliteDb
from agno.workflow import Router, Step, StepInput, Workflow

from agents.course.agent import catalog_agent
from agents.triage.agent import triage_agent
from agents.continuation.agent import continuation_agent
from settings import settings

db = SqliteDb(db_file=settings.agent_db_path)


def _select_specialist(step_input: StepInput) -> str:
    decision = step_input.previous_step_content
    return "Continuation" if getattr(decision, "agent", None) == "continuation" else "Catalog"


workflow = Workflow(
    id="doradca-szkoleniowy",
    name="Doradca szkoleniowy",
    description="Triage kieruje pytanie do Catalog albo Continuation; wybrany agent odpowiada użytkownikowi.",
    db=db,
    steps=[
        Step(name="Triage", agent=triage_agent),
        Router(
            name="Route",
            selector=_select_specialist,
            choices=[
                Step(name="Catalog", agent=catalog_agent),
                Step(name="Continuation", agent=continuation_agent),
            ],
        ),
    ],
    add_workflow_history_to_steps=True,
    num_history_runs=5,
)
