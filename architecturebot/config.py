import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv, find_dotenv


# Load .env from current or parent directories; also try package-local .env
loaded_any = False
fd = find_dotenv()
if fd:
    load_dotenv(fd, override=False)
    loaded_any = True

# Also try common locations explicitly
for candidate in [
    Path.cwd() / ".env",
    Path(__file__).resolve().parent / ".env",
    Path(__file__).resolve().parent.parent / ".env",
]:
    try:
        if candidate.exists():
            load_dotenv(str(candidate), override=False)
            loaded_any = True
    except Exception:
        pass


@dataclass
class Settings:
    telegram_bot_token: str
    openai_api_key: str
    openai_model: str
    max_tokens: int


def _get_env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None or value.strip() == "":
        raise RuntimeError(f"Environment variable '{name}' is not set")
    return value


settings = Settings(
    openai_api_key=_get_env("OPENAI_API_KEY"),
    openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "450")),
    telegram_bot_token=_get_env("TELEGRAM_BOT_TOKEN")
)


