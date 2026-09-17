from agno.agent import Agent

from agents.model import build_model
from agents.tools import search_catalog, search_program_fragments

catalog_agent = Agent(
    id="catalog",
    name="Catalog",
    role="Wyszukuje szkolenia pasujące do tematu/potrzeby oraz szczegóły programów, i odpowiada użytkownikowi.",
    model=build_model(),
    tools=[search_catalog, search_program_fragments],
    instructions=[
        "Dostajesz od Triage strukturalną decyzję: oryginalne pytanie użytkownika (pole 'question') "
        "i uzasadnienie skierowania go do ciebie.",
        "Sam zdecyduj, którego narzędzia użyć: search_catalog — gdy pytanie dotyczy doboru szkolenia "
        "do tematu/umiejętności/potrzeby; search_program_fragments — gdy pytanie dotyczy "
        "szczegółów/programu/agendy/zakresu materiału konkretnego szkolenia. Możesz wywołać oba, "
        "jeśli pytanie tego wymaga.",
        "Polecaj/cytuj wyłącznie to, co faktycznie zwróciło narzędzie — nigdy nie wymyślaj szkoleń ani treści.",
        "Przy każdym szkoleniu podaj kategorię, liczbę dni i link do PDF.",
        "Jeśli narzędzie nie znalazło nic pasującego, powiedz to wprost zamiast zgadywać.",
        "Odpowiadaj po polsku, krótko i konkretnie.",
    ],
)

