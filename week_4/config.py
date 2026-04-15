from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, ValidationError

class Settings(BaseSettings):
    anthropic_api: str = Field(..., alias="ANTHROPIC_API_KEY")
    max_tokens: int = Field(...,alias="MAX_TOKENS")
    model: str = Field(..., alias="ANTHROPIC_MODEL")
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

try :
    settings = Settings()
    print("config validated")
except ValidationError as e:
    print(e)
