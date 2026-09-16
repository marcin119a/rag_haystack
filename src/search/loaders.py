
from haystack import Document
import pandas as pd
from settings import settings
from pathlib import Path
import unicodedata
from haystack.components.preprocessors import RecursiveDocumentSplitter
import re
from dataclasses import replace
from haystack.components.converters.pypdf import PyPDFToDocument



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


def load_program_pdfs() -> list[Document]:
    courses = pd.read_parquet(settings.szkolenia_parquet)
    courses["stem"] = courses["plik"].str.removesuffix(".md")
    courses = courses.set_index("stem")

    paths = sorted(Path(settings.programy_pdf_dir).glob("*.pdf"))
    converted = PyPDFToDocument().run(sources=paths)["documents"]

    docs = []
    for path, doc in zip(paths, converted):
        row = courses.loc[unicodedata.normalize("NFC", path.stem)]
        meta = {
            "nazwa": row.nazwa,
            "kategoria": row.kategoria,
            "dni": int(row.dni),
            "pdf_url": row.pdf_url,
            "plik": row.plik,
        }
        docs.append(Document(content=doc.content, meta=meta))
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


def normalize_markdown(text: str) -> str:
    text = re.sub(r"^\*\*(#+ .+?)\*\*$", r"\1", text, flags=re.MULTILINE)
    text = text.replace(r"\.", ".")

    text = re.sub(r"(?m)^(\d+)\.\s+(.+)$", r"## \1. \2", text)
    text = re.sub(r"(?<!\n)\n(?!\n|[a-z]\.\s|##\s)", " ", text)

    return text


def split_main_sections(text: str) -> list[str]:
    sections = []
    for section in re.split(r"(?=^## \d+\.)", text, flags=re.MULTILINE):
        stripped = section.strip()
        if re.match(r"^## \d+\.", stripped):
            sections.append(stripped)
    return sections


def split_program_sections(
    docs: list[Document], split_length: int = 1200, split_overlap: int = 100
) -> list[Document]:
    splitter = RecursiveDocumentSplitter(
        split_length=split_length,
        split_overlap=split_overlap,
        split_unit="char",
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    splitter.warm_up()

    chunks: list[Document] = []
    for doc in docs:
        text = normalize_markdown(doc.content)
        for section in split_main_sections(text):
            section_doc = Document(content=section, meta=doc.meta.copy())
            if len(section) <= split_length:
                chunks.append(section_doc)
            else:
                chunks.extend(splitter.run(documents=[section_doc])["documents"])

    for i, chunk in enumerate(chunks):
        title = chunk.meta["nazwa"]
        if not chunk.content.startswith(title):
            chunks[i] = replace(chunk, id="", content=f"{title}\n\n{chunk.content}")

    return chunks