from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    openai_embedding_model: str = "text-embedding-3-small"  
    index_path: str = "data_rag/index.json"
    openai_index_path: str = "data_rag/index_openai.json"
    szkolenia_parquet: str = "data_rag/szkolenia.parquet"
    chroma_path: str = "data_rag/chroma"
    programy_dir: str = "data_rag/programy"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "szkolenia_chunki"
    embedding_model_dimension: int = 1536

settings = Settings()