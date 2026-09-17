
from typing import Literal

from agno.agent import Agent
from pydantic import BaseModel, Field

from agents.model import build_model


class Decision(BaseModel):
    question: str = Field(description="The user's question, copied exactly, without shortening or changes.")
    agent: Literal["catalog", "continuation"] = Field(
        description="The specialist agent that should handle this question."
    )
    justification: str = Field(description="One-sentence justification for the routing choice.")


triage_agent = Agent(
    id="triage",
    name="Triage",
    role="Klasyfikuje pytanie użytkownika i decyduje, do którego agenta specjalisty je skierować.",
    model=build_model(),
    output_schema=Decision,
    instructions=[
        "Twoja jedyna odpowiedzialność: przeczytać pytanie użytkownika i zdecydować, który z dwóch "
        "agentów specjalistów powinien je obsłużyć. Sam niczego nie wyszukujesz i nie odpowiadasz "
        "na pytanie merytorycznie — tylko klasyfikujesz na podstawie ogólnej intencji.",
        "catalog — domyślny wybór: pytanie o dobór szkolenia (temat, umiejętność, potrzeba) albo o "
        "szczegóły/program/agendę/zakres materiału konkretnego szkolenia.",
        "continuation — gdy użytkownik zaznacza, że już ukończył (albo jest w trakcie) konkretne "
        "szkolenie i pyta, co dalej / jakie szkolenie polecić jako kontynuację / jakie są alternatywy "
        "wobec tego, które już zna.",
        "W polu 'question' przepisz pytanie użytkownika dokładnie, jeden do jednego.",
        "Gdy pytanie pasowałoby do obu kategorii, wybierz tę najbliższą głównej intencji użytkownika.",
    ],
)
