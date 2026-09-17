
from agno.os import AgentOS
from agno.tracing import setup_tracing

from agents.course.agent import catalog_agent
from agents.triage.agent import triage_agent
from agents.workflow import db, workflow

agent_os = AgentOS(
    id="doradca-szkoleniowy-os",
    name="Doradca szkoleniowy",
    description="Triage -> Catalog/Continuation: control plane, sesje i evale w jednym miejscu.",
    db=db,
    agents=[triage_agent, catalog_agent],
    workflows=[workflow]
)

setup_tracing(db=db)

app = agent_os.get_app()

if __name__ == "__main__":
    agent_os.serve(app="agent_os:app", host="0.0.0.0", port=8000, reload=True)
