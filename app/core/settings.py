from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_name: str = "BAAI/bge-m3"
    normalize: bool = True
    max_chars: int = 2000
    trust_remote_code: bool = False
    hf_token: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )
    
def get_settings():
    return Settings()
