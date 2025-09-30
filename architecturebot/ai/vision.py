import asyncio
import base64
import imghdr
import logging
import time
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
        img.thumbnail((settings.image_max_side_px, settings.image_max_side_px))
        out = BytesIO()
        img.save(out, format="JPEG", quality=settings.image_jpeg_quality, optimize=True)
        return out.getvalue()
    except Exception:
        # If preprocessing fails, use original
        return image_bytes


# Architectural Brief knowledge base
ARCHITECTURAL_BRIEF = """
АРХИТЕКТУРНЫЙ БРИФ 3.0 — Критерии оценки

=== МАССИНГ ===
Массинг — характерная образная скульптура квартала с соблюдением ТЭП.

6 принципов формообразования:
1. Амплитудность силуэта — выразительный скайлайн, разница высот
2. Усложненная конфигурация квартала — новые городские пространства
3. Скульптурный футпринт морфотипов — уход от прямоугольников
4. Силуэтные завершения — уникальные крыши, террасы, ступени
5. «Неплоскостность» — запрет на монотонные фасады, объёмность
6. Принцип «красной собачки» — харизматичная деталь (ОПЦИОНАЛЬНО)

Ограничения:
- Эффективная этажность: 9, 16, 24, 31/32, 37/38, 48
- Террасы до 1.5 м, уступы до 0.6 м

=== ФАСАДЫ ===
Функции:
- Усиление образа массинга
- Формирование иерархии зданий
- Поддержка глобальной концепции

Требования:
- Пластика без монотонности
- Консоли: 2.5–3.0 м (НВФ), 1.5 м (префаб)
- Наклонные фасады: уклон ≤10°

=== ТЕРМИНЫ ===
- Амплитудность — разница высот для скайлайна
- Неплоскостность — объёмность, отказ от плоских поверхностей
- «Красная собачка» — харизматичная деталь (не обязательна)
- Футпринт — геометрия плана здания
- Массинг — образная скульптура квартала
"""

SYSTEM_PROMPT = (
    "Ты — опытный архитектор‑критик с экспертным взглядом. Твоя задача — дать объективную профессиональную оценку архитектуры здания, "
    "анализируя качество решений и их уместность в контексте. "
    "Пиши естественно, как пишет эксперт в своей оценке — без жёстких шаблонов и формальных заголовков. "
    "Каждое утверждение должно нести экспертную ценность, а не пересказывать очевидное. "
    "Итоговая оценка отражает архитектурную ценность проекта в целом, а не сумму галочек.\n\n"
    + ARCHITECTURAL_BRIEF
)


USER_INSTRUCTIONS = (
    "Проанализируй здание на фотографии как опытный архитектор-критик, используя критерии из Архитектурного Брифа 3.0. "
    "Дай объективную экспертную оценку в формате естественного профессионального анализа.\n\n"
    
    "Напиши критический анализ объёмом 4-6 абзацев, который включает:\n"
    "• Характеристику объекта — стилистика, образ массинга, роль в контексте\n"
    "• Оценку композиционных решений через призму принципов формообразования:\n"
    "  - Амплитудность силуэта (выразительность скайлайна)\n"
    "  - Силуэтные завершения (качество проработки верха)\n"
    "  - Неплоскостность фасада (объёмность, скульптурность, отказ от монотонности)\n"
    "  - Пропорции и масштаб (человеческий масштаб, цельность)\n"
    "  - Скульптурность футпринта (если виден план/композиция)\n"
    "  - Харизматичные детали — 'красная собачка' (ТОЛЬКО если действительно применены)\n"
    "• Анализ качества фасадных решений — материалы, пластика, деталировка, ритм\n"
    "• Особенности, которые работают на образ или, наоборот, его ослабляют\n\n"
    
    "Завершай анализ итоговой оценкой в формате:\n"
    "Архитектурная оценка: X/10\n\n"
    
    "Критерии итоговой оценки:\n"
    "• 8-10 — сильный проект: выразительный массинг, качественная пластика, цельная композиция, вклад в среду\n"
    "• 6-7 — добротная архитектура с отдельными удачными приёмами, но есть просчёты в композиции или деталировке\n"
    "• 4-5 — посредственный проект: шаблонные решения, слабая проработка массинга, монотонность\n"
    "• 1-3 — проблемный объект: серьёзные композиционные ошибки, отсутствие образа, плохое качество\n\n"
    
    "После оценки добавь:\n"
    "Общее ревью от Критика:\n"
    "Напиши 3-6 ёмких предложений с экспертным взглядом 'крутого' архитектора-критика. "
    "Подмечай профессиональные детали и нюансы, которые не видны обычному человеку: "
    "тонкости композиционных решений, скрытые приёмы, профессиональные просчёты или, наоборот, мастерские находки. "
    "Пиши остро, точно, без воды — как пишет эксперт высокого уровня, который видит суть за формой. "
    "Это должен быть концентрированный профессиональный вердикт.\n\n"
    
    "Принципы анализа:\n"
    "- Используй профессиональную терминологию из брифа (массинг, футпринт, амплитудность, неплоскостность)\n"
    "- Оценивай КАЧЕСТВО и УМЕСТНОСТЬ принципов формообразования, а не просто их наличие\n"
    "- Пиши как эксперт, который даёт объективное мнение, а не заполняет чек-лист\n"
    "- Не используй формальные заголовки типа «**Композиционное качество**» в основном тексте — пиши текстом\n"
    "- Фокусируйся на значимых для данного проекта принципах формообразования\n"
    "- 'Красная собачка' не обязательна — упоминай только если видишь яркий харизматичный акцент\n"
    "- Добавляй экспертную глубину, избегай констатации очевидного\n"
    "- Разные по качеству проекты должны получать разные оценки\n"
    "- В 'Общем ревью от Критика' покажи максимальную экспертизу — замечай то, что скрыто от непрофессионала\n"
    "- При недостатке данных отмечай это, не додумывай"
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
    image_b64 = base64.b64encode(processed).decode("utf-8")
    data_url = f"data:{mime};base64,{image_b64}"

    client = _get_client()

    async def _call() -> str:
        async def _request(model: str, instructions: str, detail: str) -> tuple[str, Optional[str]]:
            # Log outgoing request metadata (do not log raw image)
            logger.info(
                "OpenAI request: model=%s detail=%s mime=%s size_kb=%.1f image_sha256=%s instr_len=%d",
                model,
                detail,
                mime,
                len(processed) / 1024.0,
                h[:12],
                len(instructions or ""),
            )

            start_time = time.perf_counter()
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
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            finish = resp.choices[0].finish_reason if resp.choices else None
            content = resp.choices[0].message.content if resp.choices and resp.choices[0].message else None

            # Try to extract token usage if available
            usage = getattr(resp, "usage", None)
            if isinstance(usage, dict):
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
                total_tokens = usage.get("total_tokens")
            else:
                prompt_tokens = getattr(usage, "prompt_tokens", None)
                completion_tokens = getattr(usage, "completion_tokens", None)
                total_tokens = getattr(usage, "total_tokens", None)

            logger.info(
                "OpenAI response: model=%s detail=%s finish=%s tokens p/c/t=%s/%s/%s duration_ms=%.0f",
                model,
                detail,
                finish,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                duration_ms,
            )
            logger.info("OpenAI content:\n%s", (content.strip() if content else "<empty>"))
            return (content.strip() if content else ""), finish

        # 1) Primary
        text, finish = await _request(settings.openai_model, USER_INSTRUCTIONS, settings.vision_detail_primary)
        if finish == "content_filter":
            raise ValueError("Изображение не прошло проверку безопасности модели. Попробуйте другое фото фасада.")
        if text:
            _response_cache[h] = text
            return text

        logger.warning("Empty content on primary; trying fallback prompt (high detail)")
        # 2) Fallback prompt, same model, high detail
        text, finish = await _request(settings.openai_model, USER_INSTRUCTIONS_FALLBACK, settings.vision_detail_fallback)
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


