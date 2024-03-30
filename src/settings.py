from pydantic_settings import BaseSettings


class EnvironmentSettings(BaseSettings):
    LANGCHAIN_TRACING_V2: str
    LANGCHAIN_PROJECT: str
    LANGCHAIN_API_KEY: str
    OPENAI_API_KEY: str
    ANTHROPIC_API_KEY: str
    LLAMA_CLOUD_API_KEY: str
    HF_TOKEN: str

    class Config:
        env_file = ".env"


class Settings(BaseSettings):
    env: EnvironmentSettings = EnvironmentSettings()


settings = Settings()
