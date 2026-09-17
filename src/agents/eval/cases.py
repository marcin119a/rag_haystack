from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    agent_id: str  
    input: str
    expected_output: str
    guidelines: list[str] | None = None


CASES: list[Case] = [
    Case(
        agent_id="triage",
        input="Szukam szkolenia z Pythona dla początkujących.",
        expected_output=(
            "Decyzja z polem agent='catalog' (pytanie dotyczy doboru szkolenia do potrzeby, "
            "nie kontynuacji czegoś już ukończonego); pole question powtarza pytanie użytkownika "
            "bez zmian."
        ),
    ),
    Case(
        agent_id="triage",
        input="Ukończyłem szkolenie 'Docker w praktyce', co mogę zrobić dalej?",
        expected_output=(
            "Decyzja z polem agent='continuation' (użytkownik odnosi się do ukończonego "
            "szkolenia i pyta, co dalej); pole question powtarza pytanie użytkownika bez zmian."
        ),
    ),
    Case(
        agent_id="catalog",
        input="Szukam szkolenia z administracji klastrem Kubernetes.",
        expected_output=(
            "Odpowiedź poleca szkolenie związane z Kubernetesem z katalogu (np. 'Kubernetes - "
            "orkiestracja mikroserwisów' albo podobne), z podaną kategorią, liczbą dni i linkiem "
            "do PDF."
        ),
        guidelines=[
            "Nazwa realnie istniejącego szkolenia z katalogu i link do PDF muszą się pojawić w odpowiedzi.",
            "Odpowiedź jest po polsku.",
        ],
    ),
    Case(
        agent_id="catalog",
        input="Jaki jest program szkolenia Terraform na AWS?",
        expected_output=(
            "Odpowiedź opisuje zakres/program szkolenia Terraform (Infrastructure as Code na "
            "AWS), cytując realne fragmenty programu, z linkiem do PDF."
        ),
        guidelines=["Odpowiedź nie wymyśla treści programu spoza tego, co zwróciło narzędzie."],
    ),
    Case(
        agent_id="continuation",
        input="Skończyłem szkolenie 'Docker w praktyce', co polecacie jako kontynuację?",
        expected_output=(
            "Odpowiedź proponuje naturalną kontynuację związaną z Dockerem/Kubernetesem/"
            "orkiestracją kontenerów, z kategorią, liczbą dni i linkiem do PDF."
        ),
        guidelines=["Odpowiedź jest po polsku i nie wymyśla szkoleń spoza wyniku narzędzia."],
    ),
]
