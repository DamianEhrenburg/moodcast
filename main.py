import logging
import sys

from telegram import BotCommand, Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    filters,
)

from moodcast_core.config import KINOPOISK_API_KEY, TELEGRAM_BOT_TOKEN
from moodcast_core.handlers import (
    bookmark_callback,
    handle_message,
    handle_navigation,
    help_command,
    random_callback,
    random_command,
    remove_favorite_callback,
    start_command,
    type_handler,
    watchlist_callback,
    watchlist_command,
)
from moodcast_core.models import ContentType
from moodcast_core.nlp import get_emotion_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def post_init(application: Application) -> None:
    """Configures Telegram command menu autocompletion."""
    commands = [
        BotCommand("start", "Главное меню"),
        BotCommand("watchlist", "Мои закладки"),
        BotCommand("random", "Случайный хит"),
        BotCommand("help", "Справка и примеры"),
    ]
    try:
        await application.bot.set_my_commands(commands)
        logger.info("Bot commands successfully registered in Telegram menu.")
    except Exception as e:
        logger.warning(f"Could not register bot commands menu: {e}")


def main() -> None:
    if not TELEGRAM_BOT_TOKEN or not KINOPOISK_API_KEY:
        logger.critical("TELEGRAM_BOT_TOKEN or KINOPOISK_API_KEY is missing in .env. Exiting.")
        sys.exit(1)

    pipeline = get_emotion_pipeline()
    if not pipeline:
        logger.critical("Failed to load Zero-Shot NLP model. Exiting.")
        sys.exit(1)

    application = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .concurrent_updates(True)
        .post_init(post_init)
        .build()
    )

    # Commands
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler(["watchlist", "saved"], watchlist_command))
    application.add_handler(CommandHandler("random", random_command))

    # Callback queries
    application.add_handler(CallbackQueryHandler(start_command, pattern="^back_to_menu$"))
    application.add_handler(
        CallbackQueryHandler(
            type_handler,
            pattern=f"^({ContentType.MOVIE.value}|{ContentType.TV.value})$",
        )
    )
    application.add_handler(CallbackQueryHandler(random_callback, pattern="^random_hit$"))
    application.add_handler(CallbackQueryHandler(bookmark_callback, pattern="^bookmark_\\d+$"))
    application.add_handler(CallbackQueryHandler(watchlist_callback, pattern="^watchlist_\\d+$"))
    application.add_handler(CallbackQueryHandler(remove_favorite_callback, pattern="^remove_fav_\\d+_\\d+$"))
    application.add_handler(
        CallbackQueryHandler(
            handle_navigation,
            pattern="^(next|prev|like|dislike)$",
        )
    )

    # Messages
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )

    logger.info("Moodcast bot configured. Starting polling...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
