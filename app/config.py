"""
Centralized configuration — Bedrock qwen.qwen3-coder-next for ALL tasks.
Groq kept only as emergency fallback.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # AWS Bedrock
    aws_region: str = "us-east-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""

    bedrock_model_id: str = "qwen.qwen3-coder-next"
    bedrock_max_tokens: int = 8192
    bedrock_temperature: float = 0.2
    bedrock_timeout: int = 300
    bedrock_max_concurrent: int = 20

    # Groq fallback
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    groq_fast_model: str = "llama-3.1-8b-instant"
    groq_max_tokens: int = 4096
    groq_temperature: float = 0.2

    # Gemini fallback
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    # Processing
    batch_size: int = 15
    max_concurrent_llm_calls: int = 20
    max_file_size_kb: int = 500

    llm_retry_attempts: int = 3
    llm_retry_delay: float = 2.0

    # Token budgets
    max_file_tokens: int = 16000
    max_prompt_tokens: int = 32000

    # Storage
    clone_base_dir: str = "./repos"
    vector_store_dir: str = "./vector_stores"
    analysis_store_dir: str = "./analysis_data"
    cache_dir: str = "./cache"

    # RAG
    rag_top_k: int = 15
    rag_chunk_size: int = 1200
    rag_chunk_overlap: int = 200
    embedding_model: str = "all-MiniLM-L6-v2"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    # API Security
    api_secret_key: str = ""

    @property
    def clone_path(self) -> Path:
        return Path(self.clone_base_dir)

    @property
    def vector_store_path(self) -> Path:
        return Path(self.vector_store_dir)

    @property
    def analysis_store_path(self) -> Path:
        return Path(self.analysis_store_dir)

    @property
    def cache_path(self) -> Path:
        return Path(self.cache_dir)

    @property
    def gemini_available(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def bedrock_available(self) -> bool:
        return bool(self.aws_access_key_id and self.aws_secret_access_key)

    @property
    def groq_available(self) -> bool:
        return bool(self.groq_api_key)


settings = Settings()