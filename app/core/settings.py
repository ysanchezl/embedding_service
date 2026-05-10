from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_name: str = "BAAI/bge-m3"
    normalize: bool = True
    max_chars: int = 2000
    trust_remote_code: bool = False

def get_settings():
    return Settings()
