from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: SecretStr | None = None
    model: str = "claude-opus-5-5"
    effort: str = "medium"
    max_tokens: int = 16000

    chroma_dir: Path = Path("data/chroma")
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    chunk_size: int = 1000
    chunk_overlap: int = 150
    search_results: int = 6
    max_upload_mb: int = 25

    # Enable when serving over HTTPS.
    cookie_secure: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
