from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_name: str = ""
    model_dir: str = "./models"
    query_prefix: str = ""
    passage_prefix: str = ""
    normalize: bool = True
    batch_size: int = 32
    # None = derive from the model's max_seq_length
    chunk_tokens: int | None = None
    chunk_overlap_ratio: float = 0.12
    max_documents: int = 100
    max_document_chars: int = 2_000_000
    max_total_chars: int = 10_000_000
    max_pending_jobs: int = 20
    job_ttl_seconds: int = 3600
    trust_remote_code: bool = False
    hf_token: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )

def get_settings():
    return Settings()
