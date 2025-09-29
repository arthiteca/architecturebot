import logging
from typing import Optional

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from config import settings
from ai.vision import analyze_building_image
from auth.keys import get_key_info, decrement_quota, generate_keys, UNLIMITED


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("architecturebot")


WELCOME_TEXT = (
    "Привет:) Я Архитектурный критик\n"
    "Сначала отправь ключ авторизации.\n"
    "Затем пришлите фото здания — я оценю его"
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(WELCOME_TEXT)


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Сначала отправьте ключ авторизации одной строкой. Затем пришлите фото здания. "
        "Текст и голосовые я не анализирую — присылай фото или изображения."
    )


async def key_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if not message:
        return
    args = message.text.split()
    if len(args) < 2:
        await message.reply_text("Неверный ключ. Проверьте и попробуйте снова")
        return
    api_key = args[1].strip()
    info = get_key_info(api_key)
    if not info:
        await message.reply_text("Неверный ключ. Проверьте и попробуйте снова.")
        return
    # Bind key to chat in-memory per session
    context.chat_data["api_key"] = api_key
    if info.remaining == UNLIMITED:
        await message.reply_text("Ключ принят. Остаток по ключу: безлимит.")
    else:
        await message.reply_text(f"Ключ принят. Остаток изображений: {info.remaining} из 10.")


async def _download_image_bytes(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Optional[bytes]:
    message = update.effective_message
    if message is None:
        return None

    # Prioritize photos
    if message.photo:
        photo_sizes = message.photo
        best = photo_sizes[-1]  # highest resolution
        tg_file = await context.bot.get_file(best.file_id)
        data = await tg_file.download_as_bytearray()
        return bytes(data)

    # Fallback to image document
    if message.document and (message.document.mime_type or "").startswith("image/"):
        tg_file = await context.bot.get_file(message.document.file_id)
        data = await tg_file.download_as_bytearray()
        return bytes(data)

    return None


async def handle_photo_or_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id if update.effective_chat else None
    if chat_id is not None:
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    # Authorization check
    api_key = context.chat_data.get("api_key") if context.chat_data else None
    if not api_key:
        await update.effective_message.reply_text(
            "Доступ по ключу. Введите ключ."
        )
        return
    info = get_key_info(api_key)
    if not info:
        await update.effective_message.reply_text("Ключ недействителен. Попробуй еще раз.")
        return
    if info.remaining != UNLIMITED and info.remaining <= 0:
        await update.effective_message.reply_text(
            "Лимит изображений по вашему ключу исчерпан (10/10). Запросите новый ключ."
        )
        return

    image_bytes = await _download_image_bytes(update, context)
    if not image_bytes:
        await update.effective_message.reply_text(
            "Похоже, это не изображение. Пожалуйста, отправьте фото здания."
        )
        return

    status_msg = await update.effective_message.reply_text("Анализирую изображение… Это займет небольше минуты")

    try:
        result_text = await analyze_building_image(image_bytes)
        # Decrement quota only after successful analysis
        remaining = decrement_quota(api_key)
        if remaining == UNLIMITED:
            suffix = "\n\nОстаток по ключу: безлимит."
        else:
            suffix = f"\n\nОстаток по ключу: {remaining}/10."
        result_text = f"{result_text}{suffix}"
        try:
            await status_msg.edit_text(result_text)
        except Exception:
            await update.effective_message.reply_text(result_text)
    except ValueError as ve:
        try:
            await status_msg.edit_text(str(ve))
        except Exception:
            await update.effective_message.reply_text(str(ve))
    except Exception as exc:
        logger.exception("Analysis failed: %s", exc)
        msg = (
            "Сейчас наблюдается высокая нагрузка или временная ошибка. "
            "Попробуйте ещё раз чуть позже."
        )
        try:
            await status_msg.edit_text(msg)
        except Exception:
            # Fallback to plain reply
            try:
                await update.effective_message.reply_text(msg)
            except Exception:
                # Swallow to avoid crashing handler
                pass


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if not message or not message.text:
        await update.effective_message.reply_text(
            "Требуется фотография здания. Отправьте ключ одной строкой, затем фото."
        )
        return

    text = message.text.strip()
    info = get_key_info(text)
    if info:
        # Treat plain text as key; bind or replace existing
        context.chat_data["api_key"] = text
        if info.remaining == UNLIMITED:
            await message.reply_text(
                "Ключ принят (безлимит). Теперь пришлите фото здания."
            )
        else:
            await message.reply_text(
                f"Ключ принят. Остаток изображений: {info.remaining} из 10. Теперь пришлите фото здания."
            )
        return

    # Not a valid key — keep guidance concise
    if context.chat_data.get("api_key"):
        await message.reply_text(
            "Текст не анализирую. Пришлите, пожалуйста, фото здания."
        )
    else:
        await message.reply_text(
            "Это не похоже на корректный ключ. Отправьте действительный ключ одной строкой, затем фото здания."
        )


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Голосовые и текст не анализирую. Отправьте фото здания. Если ключ не введён: /key <ключ>."
    )


def main() -> None:
    application = (
        ApplicationBuilder()
        .token(settings.telegram_bot_token)
        .concurrent_updates(True)
        .build()
    )

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_cmd))
    application.add_handler(CommandHandler("key", key_cmd))
    application.add_handler(
        MessageHandler((filters.PHOTO | (filters.Document.IMAGE)), handle_photo_or_image)
    )
    application.add_handler(MessageHandler((filters.VOICE | filters.AUDIO), handle_voice))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Bot is starting…")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped.")


