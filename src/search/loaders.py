
from haystack import Document
import pandas as pd
from settings import settings
from pathlib import Path
import unicodedata
from haystack.components.preprocessors import RecursiveDocumentSplitter


def load_course_docs() -> list[Document]:
    """One document per course: name + description from the parquet. No chunking — for *_memory variants."""
    df = pd.read_parquet(settings.szkolenia_parquet)
    return [
        Document(
            content=f"{row.nazwa}\n{row.opis}",
            meta={
                "nazwa": row.nazwa,
                "kategoria": row.kategoria,
                "dni": int(row.dni),
                "pdf_url": row.pdf_url,
            },
        )
        for row in df.itertuples()
    ]


def load_program_docs() -> list[Document]:
    """One document per Markdown file (full course program); metadata from the parquet, matched by filename.
    Input to recursive_split() — for variants that chunk the programs (qdrant_hybrid, chroma_dense)."""
    courses = pd.read_parquet(settings.szkolenia_parquet).set_index("plik")
    docs = []
    for path in sorted(Path(settings.programy_dir).glob("*.md")):
        row = courses.loc[unicodedata.normalize("NFC", path.name)]
        meta = {
            "nazwa": row.nazwa,
            "kategoria": row.kategoria,
            "dni": int(row.dni),
            "pdf_url": row.pdf_url,
            "plik": path.name,
        }
        docs.append(Document(content=path.read_text(encoding="utf-8"), meta=meta))
    return docs


def recursive_split(docs: list[Document], split_length: int = 1000, split_overlap: int = 150) -> list[Document]:
    """Splits course programs into fragments along the Markdown structure (headings -> paragraphs -> sentences -> words)
    and prepends the course title to every fragment that doesn't already have it."""
    splitter = RecursiveDocumentSplitter(
        split_length=split_length,
        split_overlap=split_overlap,
        split_unit="char",
        separators=[
            "\n## ",  # Markdown sections
            "\n### ",  # subsections
            "\n\n",  # paragraphs
            "\n",  # lines
            ". ",  # sentences
            " ",  # words
            "",  # finally, individual characters
        ],
    )
    splitter.warm_up()
    chunks = splitter.run(documents=docs)["documents"]
    for chunk in chunks:
        title = chunk.meta["nazwa"]
        if not chunk.content.lstrip("# ").startswith(title):
            chunk.content = f"{title}\n\n{chunk.content}"
    return chunks