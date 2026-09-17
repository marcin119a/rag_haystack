from agno.agent import Agent

from agents.model import build_model
from agents.tools import search_related_trainings

continuation_agent = Agent(
    id="continuation",
    name="Continuation",
    role="Proponuje szkolenia jako kontynuację/alternatywę tego, które użytkownik już ukończył lub zna.",
    model=build_model(),
    tools=[search_related_trainings],
    instructions=[
        "Dostajesz od Triage strukturalną decyzję: oryginalne pytanie użytkownika (pole 'question') "
        "i uzasadnienie skierowania go do ciebie.",
        "Twoja jedyna odpowiedzialność: wywołać search_related_trainings z tematem/nazwą szkolenia, "
        "o które user już pytał lub które ukończył, i na tej podstawie zaproponować kontynuację.",
        "Polecaj/cytuj wyłącznie to, co faktycznie zwróciło narzędzie — nigdy nie wymyślaj szkoleń ani treści.",
        "Przy każdym szkoleniu podaj kategorię, liczbę dni i link do PDF.",
        "Jeśli narzędzie nie znalazło nic pasującego, powiedz to wprost zamiast zgadywać.",
        "Odpowiadaj po polsku, krótko i konkretnie.",
    ],
)
