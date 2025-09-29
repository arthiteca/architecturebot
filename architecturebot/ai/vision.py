import asyncio
import base64
import imghdr
import logging
from typing import Optional

from openai import OpenAI, PermissionDeniedError

from config import settings


logger = logging.getLogger(__name__)


_client: Optional[OpenAI] = None
_response_cache: dict[str, str] = {}


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def _detect_mime_type(image_bytes: bytes) -> str:
    kind = imghdr.what(None, h=image_bytes)
    if kind == "jpeg":
        return "image/jpeg"
    if kind == "png":
        return "image/png"
    if kind == "webp":
        return "image/webp"
    # Fallback
    return "application/octet-stream"


def _preprocess_image(image_bytes: bytes) -> bytes:
    try:
        from io import BytesIO
        from PIL import Image

        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        img.thumbnail((1024, 1024))
        out = BytesIO()
        img.save(out, format="JPEG", quality=80, optimize=True)
        return out.getvalue()
    except Exception:
        # If preprocessing fails, use original
        return image_bytes


SYSTEM_PROMPT = (
    "Ты — архитектор‑критик. Отвечай лаконично, чётко, профессионально и объективно. "
    "Без воды, без эмоциональных оценок, без повторов. Только проверяемые наблюдения по изображению. "
    "Оценивай прежде всего КАЧЕСТВО и УМЕСТНОСТЬ приёмов, их вклад в цельность композиции, человеческий масштаб и контекст — а не просто наличие. "
    "‘Красная собачка’ опциональна: отсутствие не считается минусом по умолчанию. Если данных недостаточно — помечай ‘неопределимо’. Строго соблюдай структуру ответа."
)


USER_INSTRUCTIONS = (
    "Проанализируй фото здания и выдай краткую критическую оценку по структуре ниже. "
    "Стиль: деловой, сдержанный, объективный. Краткость обязательна: 3–4 абзаца суммарно, в каждом блоке 1–2 ёмких предложения, без общих фраз.\n\n"
    "Архитектурная оценка здания:\n"
    "[Стиль и образ]\n"
    "[Фасадные решения]\n"
    "[Контекст и функциональность]\n"
    "[Инновационность / исторический контекст (если есть)]\n\n"
    "[Принципы формообразования]\n"
    "1) Амплитудность силуэта — статус (соблюдён / частично / нарушен / неопределимо) + один короткий довод (оценка качества и уместности)\n"
    "2) Выразительность силуэта — статус + один довод (качество/влияние)\n"
    "3) Пропорционирование — статус + один довод (масштаб/цельность)\n"
    "4) Завершения — статус + один довод (прочтение верха)\n"
    "5) Неплоскостность — статус + один довод (объём/скульптурность без перегруза)\n"
    "6) Красная собачка — статус + один довод (ОПЦИОНАЛЬНО; отсутствие не минус, если проект сознательно сдержанный)\n\n"
    "Архитектурная оценка: X/10\n"
    "[Краткое резюме: 1–2 фразы — сильные стороны и конкретные улучшения]\n\n"
    "Указания:\n"
    "- Оценивай не ‘факт наличия’, а качество исполнения и уместность приёмов и их вклад в цельность и контекст.\n"
    "- ‘Красная собачка’ не обязательна; фиксируй как плюс только если действительно уместна и работает.\n"
    "- Общая оценка X/10 — не сумма статусов; учитывай цельность композиции, масштаб и ритм, качество деталировки/материалов, читаемость первых этажей и влияние на среду. Избегай одинаковых оценок без оснований.\n"
    "- Если данных недостаточно — помечай ‘неопределимо’, без домыслов."
)

# Minimal fallback prompt for cases when the first reply is empty or filtered
USER_INSTRUCTIONS_FALLBACK = (
    "Кратко и по делу опиши архитектуру здания на фото: стиль/образ, фасад (материалы, ритм), контекст, итоговая оценка 1–10. "
    "Не более 3–4 коротких предложений."
)


async def analyze_building_image(image_bytes: bytes) -> str:
    """Send image to OpenAI Vision-capable model and return formatted assessment text."""

    if not image_bytes or len(image_bytes) < 10_000:
        raise ValueError("Недостаточно данных изображения. Попробуйте отправить фото в более высоком качестве.")

    processed = _preprocess_image(image_bytes)
    # Cache by hash of processed image
    import hashlib
    h = hashlib.sha256(processed).hexdigest()
    if h in _response_cache:
        return _response_cache[h]

    mime = _detect_mime_type(processed)
    image_b64 = base64.b64encode(processed).decode("ascii")
    data_url = f"data:{mime};base64,{image_b64}"

    client = _get_client()

    async def _call() -> str:
        async def _request(model: str, instructions: str, detail: str) -> tuple[str, Optional[str]]:
            resp = await asyncio.to_thread(
                client.chat.completions.create,
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": instructions},
                            {"type": "image_url", "image_url": {"url": data_url, "detail": detail}},
                        ],
                    },
                ],
                timeout=60,
            )
            finish = resp.choices[0].finish_reason if resp.choices else None
            content = resp.choices[0].message.content if resp.choices and resp.choices[0].message else None
            return (content.strip() if content else ""), finish

        # 1) Primary
        text, finish = await _request(settings.openai_model, USER_INSTRUCTIONS, "low")
        if finish == "content_filter":
            raise ValueError("Изображение не прошло проверку безопасности модели. Попробуйте другое фото фасада.")
        if text:
            _response_cache[h] = text
            return text

        logger.warning("Empty content on primary; trying fallback prompt (high detail)")
        # 2) Fallback prompt, same model, high detail
        text, finish = await _request(settings.openai_model, USER_INSTRUCTIONS_FALLBACK, "high")
        if text:
            _response_cache[h] = text
            return text

        # 3) Model fallback
        alt_model = "gpt-4o" if "mini" in settings.openai_model else "gpt-4o-mini"
        logger.warning("Empty content on fallback prompt; trying alternate model: %s", alt_model)
        text, finish = await _request(alt_model, USER_INSTRUCTIONS_FALLBACK, "high")
        if text:
            _response_cache[h] = text
            return text

        return "Не удалось сформировать оценку. Попробуйте другое фото (фронтальный фасад, без бликов и шума)."

    # Exponential backoff on transient / rate limit errors
    delay_seconds = 1.0
    last_exc: Optional[Exception] = None
    for attempt in range(4):
        try:
            return await asyncio.wait_for(_call(), timeout=60)
        except asyncio.TimeoutError:
            if attempt < 3:
                logger.warning("OpenAI call timeout (attempt %d)", attempt + 1)
                await asyncio.sleep(delay_seconds)
                delay_seconds *= 2
                continue
            raise ValueError("Превышено время ожидания ответа модели. Попробуйте ещё раз позже.")
        except PermissionDeniedError as exc:
            message = str(exc)
            # Specific handling for region/account restrictions
            if "unsupported_country_region_territory" in message.lower() or "not supported" in message.lower():
                logger.error("OpenAI access denied due to region/account restrictions: %s", message)
                raise ValueError(
                    "К сожалению, доступ к модели ограничен для вашего региона или аккаунта. "
                    "Варианты: использовать аккаунт/организацию с поддерживаемым регионом, "
                    "либо Azure OpenAI в разрешённом регионе, либо обратиться в поддержку OpenAI."
                ) from exc
            logger.exception("OpenAI permission denied: %s", message)
            raise
        except Exception as exc:  # Broad except to be robust across client versions
            last_exc = exc
            message = str(exc)
            is_rate = "rate" in message.lower() or "429" in message
            is_transient = any(word in message.lower() for word in [
                "timeout", "temporarily", "unavailable", "connection", "retry",
            ])
            if attempt < 3 and (is_rate or is_transient):
                logger.warning("OpenAI call failed (attempt %d): %s", attempt + 1, message)
                await asyncio.sleep(delay_seconds)
                delay_seconds *= 2
                continue
            logger.exception("OpenAI Vision call failed: %s", message)
            raise

    # Should not reach here
    raise RuntimeError("Не удалось получить ответ от модели. Попробуйте позже.")


