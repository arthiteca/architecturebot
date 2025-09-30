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
    openai_timeout_seconds: int
    openai_max_retries: int
    openai_backoff_base_seconds: float
    image_max_side_px: int
    image_jpeg_quality: int
    vision_detail_primary: str
    vision_detail_fallback: str


def _get_env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None or value.strip() == "":
        raise RuntimeError(f"Environment variable '{name}' is not set")
    return value


settings = Settings(
    openai_api_key=_get_env("OPENAI_API_KEY"),
    openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "450")),
    openai_timeout_seconds=int(os.getenv("OPENAI_TIMEOUT", "60")),
    openai_max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "4")),
    openai_backoff_base_seconds=float(os.getenv("OPENAI_BACKOFF_BASE", "1.0")),
    image_max_side_px=int(os.getenv("IMAGE_MAX_SIDE", "896")),
    image_jpeg_quality=int(os.getenv("IMAGE_JPEG_QUALITY", "82")),
    vision_detail_primary=os.getenv("VISION_DETAIL_PRIMARY", "low"),
    vision_detail_fallback=os.getenv("VISION_DETAIL_FALLBACK", "low"),
    telegram_bot_token=_get_env("TELEGRAM_BOT_TOKEN")
)


