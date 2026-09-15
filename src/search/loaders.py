
from haystack import Document
import pandas as pd
from settings import settings


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
