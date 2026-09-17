"""Indexer wariantu neo4j_neighbors: nie ma własnego indeksu — współdzieli z neo4j_semantic ten sam
graf w Neo4j (węzły :Szkolenie, embeddingi, fulltext index, krawędzie :PODOBNE_DO). Oba warianty
odpytują tę samą bazę na dwa różne sposoby (patrz ich searcher.py), więc budowanie/sprawdzanie
gotowości indeksu deleguje się w całości do Neo4jSemanticIndexer — `uv run search index
neo4j_neighbors` i `uv run search index neo4j_semantic` robią dokładnie to samo."""

from __future__ import annotations

from search.variants.neo4j_semantic.indexer import Neo4jSemanticIndexer, build_indexer

__all__ = ["Neo4jSemanticIndexer", "build_indexer"]
