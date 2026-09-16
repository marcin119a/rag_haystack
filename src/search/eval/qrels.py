from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Query:
    text: str
    relevant: dict[str, int]

# @TODO zapytania semantyczne

QUERIES: list[Query] = [
    Query(
        "szkolenie z Dockera dla początkujących",
        {"Docker w praktyce": 2},
    ),
    Query(
        "jak administrować klastrem Kubernetes",
        {
            "Kubernetes - orkiestracja mikroserwisów": 2,
            "Kubernetes w praktyce": 2,
            "Certyfikowany Administrator Kubernetesa": 1,
            "Kubernetes szybki start (dla deweloperów)": 1,
        },
    ),
    Query(
        "kurs Pythona dla początkujących",
        {"Python podstawy": 2},
    ),
    Query(
        "uczenie maszynowe w Pythonie",
        {
            "Uczenie maszynowe w języku Python - warsztat profesjonalisty": 2,
            "Python - uczenie maszynowe - wprowadzenie": 2,
        },
    ),
    Query(
        "testy penetracyjne aplikacji webowych",
        {"Testy penetracyjne aplikacji internetowych": 2},
    ),
    Query(
        "wprowadzenie do języka SQL",
        {"Wprowadzenie do języka SQL i bazy PostgreSQL": 2},
    ),
    Query(
        "zarządzanie projektami metodą Scrum",
        {"Zarządzanie projektami zgodnie z SCRUM": 2, "Scrum Product Owner": 1},
    ),
    Query(
        "budowa systemów RAG z wykorzystaniem dużych modeli językowych",
        {
            "Retrieval Augmented Generation (RAG) - systemy AI do wyszukiwania informacji": 2,
            "Od API LLM do Agent-RAG – budowa systemów RAG": 2,
        },
    ),
    Query(
        "bezpieczeństwo interfejsów API",
        {"Bezpieczeństwo API": 2, "Bezpieczeństwo aplikacji dla deweloperów": 1},
    ),
    Query(
        "tworzenie aplikacji mobilnych na iOS",
        {"iOS - podstawy tworzenia aplikacji": 2},
    ),
    Query(
        "automatyzacja testów w Selenium z Javą",
        {"Automatyzacja testów funkcjonalnych aplikacji internetowych z użyciem Selenium/Java": 2},
    ),
    Query(
        "podstawy kontroli wersji Git",
        {"Kontrola wersji z Git": 2},
    ),
    Query(
        "administracja systemem Linux",
        {"Praca i administracja systemem Linux": 2, "Zaawansowana administracja systemem Linux": 1},
    ),
    Query(
        "wzorce projektowe w języku C#",
        {"Wzorce projektowe w C#": 2},
    ),
    Query(
        "Infrastructure as Code z Terraform w AWS",
        {"Terraform - automatyzacja wdrożeń Infrastructure as Code w chmurze AWS": 2},
    ),
    Query(
        "prompt engineering dla modeli językowych",
        {"Prompt engineering": 2},
    ),
    Query(
        "przejście od architektury monolitycznej do mikroserwisów",
        {
            "Od monolitu do mikroserwisów - przegląd podejść architektonicznych": 2,
            "Architektura Mikroserwisów na platformie Java": 1,
        },
    ),
    Query(
        "certyfikacja ISTQB dla testerów oprogramowania",
        {
            "ISTQB® Certified Tester Foundation Level": 2,
            "Egzamin ISTQB® Certified Tester Foundation Level": 2,
        },
    ),
    Query(
        "wizualizacja danych w Power BI",
        {"Podstawy Power BI - wizualizacja danych": 2},
    ),
    Query(
        "LangGraph",
        {"Retrieval Augmented Generation (RAG) - systemy AI do wyszukiwania informacji": 2},
    ),
]
