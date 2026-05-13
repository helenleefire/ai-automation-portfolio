import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, ValidationError

_ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.env")

class Setting(BaseSettings):
    anthropic_api: str = Field(..., alias="ANTHROPIC_API_KEY")
    max_tokens: int = Field(..., alias="MAX_TOKENS")
    model: str = Field(..., alias="ANTHROPIC_MODEL")
    model_config=SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore"
    )
try:
    setting = Setting()
except ValidationError as e:
    print(e)