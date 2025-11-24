"""Centralised configuration using environment variables."""

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration aware of env vars and .env files."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    mongodb_uri: str = Field(default="mongodb+srv://<user>:<password>@<cluster>/?retryWrites=true&w=majority")
    mongodb_db_name: str = Field(default="sample_mflix")
    mongodb_collection_name: str = Field(default="movies")
    mongodb_vector_index_name: str = Field(default="kb_documents_vs_idx")

    use_mock_db: bool = Field(default=False)
    use_fake_embeddings: bool = Field(default=False)

    openai_api_key: str = Field(default="")
    embedding_model_name: str = Field(default="text-embedding-3-small")

    default_top_k: int = Field(default=5)
    num_candidates: int = Field(default=50)

    langsmith_api_key: str = Field(default="")
    langsmith_project: str = Field(default="")
    langsmith_run_name: str = Field(default="doc-search")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings so every module shares the same instance."""

    return Settings()
