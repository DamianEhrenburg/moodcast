import io
import logging
from typing import Dict, Optional

import aiohttp
import telegram.error
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from .config import DEFAULT_POSTER_URL
from .database import db
from .intent import parse_user_intent
from .kinopoisk import KinopoiskQuotaExceededError, RecommendationEngine
from .models import ContentType, EmotionalState, UserState, user_states
from .nlp import EmotionAnalyzer
from .utils import escape_html

logger = logging.getLogger(__name__)

QUOTA_EXCEEDED_MESSAGE = (
    "⚠️ <b>Каталог фильмов временно перегружен</b>\n\n"
    "Достигнут суточный лимит обращений к базе данных Кинопоиска. "
    "Доступ автоматически возобновится в полночь (00:00 UTC).\n\n"
    "Пока вы можете посмотреть ранее сохранённые фильмы в разделе <b>/watchlist</b>."
)

emotion_analyzer = EmotionAnalyzer()
recommendation_engine = RecommendationEngine()


def get_start_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("Фильмы", callback_data=ContentType.MOVIE.value),
            InlineKeyboardButton("Сериалы", callback_data=ContentType.TV.value),
        ],
        [
            InlineKeyboardButton("Случайный хит", callback_data="random_hit"),
            InlineKeyboardButton("Мои закладки", callback_data="watchlist_1"),
        ],
    ])


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    logger.info(f"User {user.id} ({user.first_name}) started the bot.")
    user_states[user.id] = UserState(user_id=user.id)

    greeting = (
        f"Привет, {escape_html(user.first_name)}.\n\n"
        "Подбираю фильмы и сериалы под ваше настроение или конкретные сюжеты.\n"
        "Выберите формат:"
    )
    if update.callback_query:
        await update.callback_query.answer()
        await update.callback_query.edit_message_text(greeting, reply_markup=get_start_keyboard())
    else:
        await update.message.reply_text(greeting, reply_markup=get_start_keyboard())


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_text = (
        "<b>Как пользоваться Moodcast:</b>\n\n"
        "1. Выберите формат (фильмы или сериалы).\n"
        "2. Напишите настроение или тему свободными словами.\n"
        "   <i>Примеры:\n"
        "   • «Загадочный триллер про космос и выживание»\n"
        "   • «Хочу что-то весёлое и лёгкое после тяжёлого дня»\n"
        "   • «Глубокая драма про музыкантов»\n"
        "   • «Страшный хоррор про заброшенные места»</i>\n\n"
        "3. Оценивайте рекомендации и сохраняйте интересное в закладки.\n"
        "4. Нажмите «Случайный хит», если не знаете, что выбрать.\n\n"
        "<b>Команды бота:</b>\n"
        "/start — Главное меню\n"
        "/watchlist — Ваши сохранённые фильмы\n"
        "/random — Случайный хит\n"
        "/help — Справка"
    )
    await update.message.reply_text(help_text, parse_mode="HTML")


async def type_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    user_id = update.effective_user.id
    user_state = user_states.setdefault(user_id, UserState(user_id=user_id))

    selected_type = query.data
    user_state.type_selected = selected_type
    user_state.reset_recommendations()

    content_rus = "сериалы" if selected_type == ContentType.TV.value else "фильмы"
    logger.info(f"User {user_id} selected: {selected_type}")

    text = (
        f"Ищем {content_rus}.\n\n"
        "Напишите настроение, жанр или тему (например: <i>«динамичный боевик про ограбление»</i>):"
    )
    try:
        await query.edit_message_text(text=text, parse_mode="HTML")
    except telegram.error.BadRequest:
        await context.bot.send_message(chat_id=user_id, text=text, parse_mode="HTML")


async def random_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_state = user_states.setdefault(user_id, UserState(user_id=user_id))
    content_type = user_state.type_selected

    wait_msg = await update.message.reply_text("Выбираю хит с высоким рейтингом...")

    try:
        details = await recommendation_engine.get_random_hit(
            content_type=content_type,
            excluded_ids=user_state.shown_recommendations,
        )
    except KinopoiskQuotaExceededError:
        try:
            await wait_msg.delete()
        except Exception:
            pass
        await update.message.reply_text(QUOTA_EXCEEDED_MESSAGE, parse_mode="HTML", disable_web_page_preview=True)
        return

    try:
        await wait_msg.delete()
    except Exception:
        pass

    if not details:
        await update.message.reply_text("Не удалось загрузить фильм. Попробуйте еще раз.")
        return

    await _send_single_movie_card(update, context, details)


async def random_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    # Silent answer stops loading spinner without displaying any black toast banner
    await query.answer()
    user_id = update.effective_user.id
    user_state = user_states.setdefault(user_id, UserState(user_id=user_id))
    content_type = user_state.type_selected

    try:
        details = await recommendation_engine.get_random_hit(
            content_type=content_type,
            excluded_ids=user_state.shown_recommendations,
        )
    except KinopoiskQuotaExceededError:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=QUOTA_EXCEEDED_MESSAGE,
            parse_mode="HTML",
            disable_web_page_preview=True,
        )
        return

    if not details:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Не удалось загрузить фильм. Попробуйте позже.",
        )
        return

    await _send_single_movie_card(update, context, details)


async def _send_photo_with_fallback(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    poster_url: Optional[str],
    caption: str,
    reply_markup: InlineKeyboardMarkup,
    alt_poster_url: Optional[str] = None,
) -> Optional[telegram.Message]:
    """
    Tries sending photo by URL, then falls back to downloading image bytes locally
    via aiohttp and uploading to Telegram, then tries fallback URLs, and only as
    a last resort falls back to send_message.
    """
    # 1. Direct URL send
    if poster_url and not poster_url.startswith("http://localhost"):
        try:
            return await context.bot.send_photo(
                chat_id=chat_id,
                photo=poster_url,
                caption=caption,
                parse_mode="HTML",
                reply_markup=reply_markup,
            )
        except Exception as e:
            logger.warning(f"Direct photo send failed for {poster_url}: {e}. Retrying via local download...")

    # 2. Local download & stream upload fallback
    candidate_urls = [u for u in [poster_url, alt_poster_url, DEFAULT_POSTER_URL] if u]
    for url in candidate_urls:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=7)) as resp:
                    if resp.status == 200:
                        img_bytes = await resp.read()
                        if img_bytes and len(img_bytes) > 200:
                            bio = io.BytesIO(img_bytes)
                            bio.name = "poster.jpg"
                            return await context.bot.send_photo(
                                chat_id=chat_id,
                                photo=bio,
                                caption=caption,
                                parse_mode="HTML",
                                reply_markup=reply_markup,
                            )
        except Exception as e:
            logger.warning(f"Download fallback failed for {url}: {e}")

    # 3. Final fallback: text message
    logger.error("All photo delivery attempts failed. Falling back to text message.")
    return await context.bot.send_message(
        chat_id=chat_id,
        text=caption,
        parse_mode="HTML",
        reply_markup=reply_markup,
        disable_web_page_preview=True,
    )


async def _send_single_movie_card(update: Update, context: ContextTypes.DEFAULT_TYPE, details: Dict) -> None:
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    user_state = user_states.setdefault(user_id, UserState(user_id=user_id))
    film_id = details.get("kinopoiskId")

    caption = _build_caption(details)
    is_saved = db.is_in_watchlist(user_id, film_id)
    bookmark_text = "✓ В закладках" if is_saved else "📌 В закладки"

    keyboard = [
        [
            InlineKeyboardButton(bookmark_text, callback_data=f"bookmark_{film_id}"),
            InlineKeyboardButton("🎲 Другой хит", callback_data="random_hit"),
        ],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    poster_url = details.get("posterUrlPreview") or details.get("posterUrl") or DEFAULT_POSTER_URL
    alt_poster_url = details.get("posterUrl") if poster_url != details.get("posterUrl") else None

    # Delete previous card to keep the chat clean
    old_msg_id = user_state.last_message_id
    if not old_msg_id and update.callback_query and update.callback_query.message:
        old_msg_id = update.callback_query.message.message_id

    if old_msg_id:
        try:
            await context.bot.delete_message(chat_id, old_msg_id)
        except Exception:
            pass
        finally:
            user_state.last_message_id = None

    new_msg = await _send_photo_with_fallback(
        context=context,
        chat_id=chat_id,
        poster_url=poster_url,
        caption=caption,
        reply_markup=reply_markup,
        alt_poster_url=alt_poster_url,
    )

    if new_msg:
        user_state.last_message_id = new_msg.message_id
        if film_id:
            user_state.add_shown_recommendation(film_id)


async def watchlist_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    await _render_watchlist_page(update, context, user_id, page=1)


async def watchlist_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    user_id = update.effective_user.id
    parts = query.data.split("_")
    page = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
    await _render_watchlist_page(update, context, user_id, page=page, edit_message=True)


async def _render_watchlist_page(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    user_id: int,
    page: int = 1,
    edit_message: bool = False,
) -> None:
    page_size = 5
    items, total_count = db.get_user_watchlist(user_id, page=page, page_size=page_size)

    if total_count == 0:
        msg = "У вас пока нет сохранённых фильмов.\n\nНажмите кнопку «В закладки» при просмотре карточки любого фильма."
        if edit_message and update.callback_query:
            await update.callback_query.edit_message_text(
                msg, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Главное меню", callback_data="back_to_menu")]])
            )
        else:
            await update.message.reply_text(msg)
        return

    total_pages = (total_count + page_size - 1) // page_size
    text_lines = [f"📌 <b>Ваши закладки (стр. {page}/{total_pages}, всего: {total_count}):</b>\n"]

    keyboard = []
    for idx, item in enumerate(items, 1):
        name = escape_html(item["name_ru"])
        year = item["year"] or ""
        rating = item["rating"] or "N/A"
        f_id = item["film_id"]
        watched_icon = "✅" if item["is_watched"] else "⏳"
        web_url = item["web_url"] or f"https://www.kinopoisk.ru/film/{f_id}/"

        text_lines.append(
            f"{watched_icon} <b>{name}</b> ({year}) — ⭐ {rating}\n"
            f"   <a href=\"{escape_html(web_url)}\">Открыть на Кинопоиске</a>"
        )
        keyboard.append([
            InlineKeyboardButton(f"❌ Удалить «{name[:20]}»", callback_data=f"remove_fav_{f_id}_{page}")
        ])

    nav_row = []
    if page > 1:
        nav_row.append(InlineKeyboardButton("⬅️ Назад", callback_data=f"watchlist_{page - 1}"))
    if page < total_pages:
        nav_row.append(InlineKeyboardButton("Вперёд ➡️", callback_data=f"watchlist_{page + 1}"))

    if nav_row:
        keyboard.append(nav_row)

    keyboard.append([InlineKeyboardButton("🏠 Главное меню", callback_data="back_to_menu")])
    reply_markup = InlineKeyboardMarkup(keyboard)

    message_text = "\n\n".join(text_lines)
    if edit_message and update.callback_query:
        try:
            await update.callback_query.edit_message_text(
                message_text, parse_mode="HTML", reply_markup=reply_markup, disable_web_page_preview=True
            )
        except telegram.error.BadRequest:
            await context.bot.send_message(
                user_id, message_text, parse_mode="HTML", reply_markup=reply_markup, disable_web_page_preview=True
            )
    else:
        await update.message.reply_text(
            message_text, parse_mode="HTML", reply_markup=reply_markup, disable_web_page_preview=True
        )


async def remove_favorite_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    user_id = update.effective_user.id
    parts = query.data.split("_")
    film_id = int(parts[2])
    page = int(parts[3]) if len(parts) > 3 else 1

    db.remove_from_watchlist(user_id, film_id)
    await query.answer("Удалено из закладок 🗑️", show_alert=False)
    await _render_watchlist_page(update, context, user_id, page=page, edit_message=True)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    user_id = user.id
    text = update.message.text
    if not text:
        return

    user_state = user_states.setdefault(user_id, UserState(user_id=user_id))

    intent = parse_user_intent(text)

    if intent.content_type_override:
        user_state.type_selected = intent.content_type_override
    elif not user_state.type_selected:
        user_state.type_selected = ContentType.MOVIE.value

    if not user_state.can_make_request():
        await update.message.reply_text("⏳ Подождите немного перед следующим запросом.")
        return
    user_state.update_request_time()

    user_state.add_message(text)
    logger.info(f"User {user_id} query: '{text}' (Intent: {intent})")

    thinking_msg = await update.message.reply_text("Ищу подходящие варианты...")
    user_state.reset_recommendations()

    try:
        emotional_state = await emotion_analyzer.analyze(text)
        if not emotional_state:
            emotional_state = EmotionalState(primary_emotion="neutral", intensity=0.0)

        user_state.emotional_state = emotional_state
        logger.info(
            f"User {user_id} emotion: {emotional_state.primary_emotion} "
            f"(Score: {emotional_state.intensity:.3f})"
        )

        recommendations = await recommendation_engine.get_recommendations(
            emotional_state,
            user_state.type_selected,
            user_state,
            topic_keyword=intent.topic_keyword,
            user_prompt=text,
            intent=intent,
        )

        if not recommendations:
            content_rus = "сериалов" if user_state.type_selected == ContentType.TV.value else "фильмов"
            if intent.explicit_genres:
                msg = (
                    f"По запросу с жанром «{', '.join(intent.explicit_genres)}» ничего не нашлось среди {content_rus}. "
                    "Попробуйте уточнить формулировку или выбрать другой формат (/start)."
                )
            elif intent.topic_keyword:
                msg = (
                    f"По теме «{intent.topic_keyword}» ничего не нашлось среди {content_rus}. "
                    "Попробуйте изменить формулировку или выбрать другой формат (/start)."
                )
            else:
                msg = (
                    "По вашему запросу ничего не нашлось. "
                    "Попробуйте описать настроение иначе или начните заново (/start)."
                )
            await thinking_msg.edit_text(msg)
            return

        user_state.recommendations = recommendations
        user_state.current_index = 0

        try:
            await thinking_msg.delete()
        except Exception:
            pass

        await send_recommendation(update, context)

    except KinopoiskQuotaExceededError:
        logger.warning(f"Kinopoisk API quota exceeded for user {user_id}")
        await thinking_msg.edit_text(QUOTA_EXCEEDED_MESSAGE, parse_mode="HTML", disable_web_page_preview=True)
        return
    except Exception as e:
        logger.error(f"Error handling message for {user_id}: {e}", exc_info=True)
        try:
            await thinking_msg.edit_text(
                "Произошла ошибка при поиске. Пожалуйста, попробуйте позже."
            )
        except Exception:
            pass


def _build_caption(details: Dict) -> str:
    film_id = details.get("kinopoiskId")
    film_name_esc = escape_html(details.get("nameRu") or details.get("nameOriginal") or "Фильм")
    original_name_esc = escape_html(details.get("nameOriginal") or "")
    desc_raw = details.get("description")
    desc_text = desc_raw.strip() if desc_raw is not None else "Описание отсутствует."
    desc_esc = escape_html(desc_text)

    kinopoisk_url = details.get("webUrl") or f"https://www.kinopoisk.ru/film/{film_id}/"
    rating_kinopoisk = details.get("ratingKinopoisk") or details.get("rating") or "N/A"
    rating_imdb = details.get("ratingImdb") or "N/A"
    year = details.get("year") or "N/A"
    countries = escape_html(", ".join(c["country"] for c in details.get("countries", [])) or "N/A")
    genres = escape_html(", ".join(g["genre"] for g in details.get("genres", [])) or "N/A")

    duration_str = details.get("filmLengthFormatted")
    seasons_count = details.get("seasonsCount")
    duration_info = ""
    if details.get("type") == "TV_SERIES" and seasons_count is not None:
        season_count_display = str(seasons_count) if seasons_count != "N/A" else "?"
        duration_info = f"📺 Сезонов: {season_count_display}"
    elif duration_str:
        duration_info = f"⏱ {duration_str}"

    type_icon = "📺" if details.get("type") == "TV_SERIES" else "🎬"
    caption_parts = [
        f"{type_icon} <b>{film_name_esc}</b>",
        f"<i>{original_name_esc}</i>" if original_name_esc and original_name_esc.lower() != film_name_esc.lower() else "",
        f"\n⭐ Кинопоиск: {rating_kinopoisk} | IMDb: {rating_imdb}",
        f"📅 Год: {year}",
        duration_info,
        f"🌍 Страна: {countries}",
        f"🎭 Жанры: {genres}",
        f"\n{desc_esc}",
        f'\n<a href="{escape_html(kinopoisk_url)}">Подробнее на Кинопоиске</a>',
    ]
    caption = "\n".join(filter(None, caption_parts))

    MAX_CAPTION_LENGTH = 1024
    if len(caption) > MAX_CAPTION_LENGTH:
        logger.warning(f"Caption for ID {film_id} is too long ({len(caption)}), truncating.")
        link_part = f'<a href="{escape_html(kinopoisk_url)}">Подробнее...</a>'
        base_length = sum(len(part) for i, part in enumerate(caption_parts) if i not in [7, 8])
        available_desc_len = MAX_CAPTION_LENGTH - base_length - len(link_part) - 20

        if available_desc_len < 50:
            truncated_description = "<i>(Описание слишком длинное для отображения)</i>"
        else:
            cutoff_pos = desc_esc.rfind(".", 0, available_desc_len)
            if cutoff_pos == -1 or cutoff_pos < available_desc_len // 2:
                cutoff_pos = available_desc_len
            else:
                cutoff_pos += 1
            truncated_description = desc_esc[:cutoff_pos] + ("..." if len(desc_esc) > cutoff_pos else "")

        caption_parts[7] = f"\n{truncated_description}"
        caption_parts[8] = f"\n{link_part}"
        caption = "\n".join(filter(None, caption_parts))

        if len(caption) > MAX_CAPTION_LENGTH:
            caption = caption[:MAX_CAPTION_LENGTH - len(link_part) - 5] + "...\n" + link_part

    return caption


async def send_recommendation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_state = user_states.get(user_id)

    if not user_state or not user_state.recommendations:
        return

    chat_id = update.effective_chat.id
    details: Optional[Dict] = None
    film_id: Optional[int] = None

    # Find next recommendation with valid description and details
    while user_state.current_index < len(user_state.recommendations):
        candidate = user_state.recommendations[user_state.current_index]
        cand_id = candidate.get("kinopoiskId")
        if not cand_id:
            user_state.current_index += 1
            continue

        try:
            async with aiohttp.ClientSession() as session:
                fetched = await recommendation_engine.get_film_details(cand_id, session)
        except KinopoiskQuotaExceededError:
            await context.bot.send_message(
                chat_id=chat_id,
                text=QUOTA_EXCEEDED_MESSAGE,
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
            return

        if not fetched:
            fetched = candidate

        desc = fetched.get("description") or fetched.get("shortDescription")
        if (desc and len(desc.strip()) >= 20) or user_state.current_index == len(user_state.recommendations) - 1:
            details = fetched
            film_id = cand_id
            user_state.recommendations[user_state.current_index] = details
            break

        user_state.current_index += 1

    if not details or not film_id:
        return

    caption = _build_caption(details)
    poster_url = details.get("posterUrlPreview") or details.get("posterUrl") or DEFAULT_POSTER_URL
    alt_poster_url = details.get("posterUrl") if poster_url != details.get("posterUrl") else None

    is_saved = db.is_in_watchlist(user_id, film_id)
    bookmark_icon = "✓ В закладках" if is_saved else "📌 В закладки"

    keyboard_rows = [
        [
            InlineKeyboardButton("👍", callback_data="like"),
            InlineKeyboardButton("👎", callback_data="dislike"),
            InlineKeyboardButton(bookmark_icon, callback_data=f"bookmark_{film_id}"),
        ]
    ]

    nav_row = []
    if user_state.history_index > 0:
        nav_row.append(InlineKeyboardButton("⬅️ Назад", callback_data="prev"))
    if user_state.current_index < len(user_state.recommendations) - 1 or user_state.history_index < len(user_state.shown_history) - 1:
        nav_row.append(InlineKeyboardButton("Далее ➡️", callback_data="next"))
    if nav_row:
        keyboard_rows.append(nav_row)

    reply_markup = InlineKeyboardMarkup(keyboard_rows)

    if user_state.last_message_id:
        try:
            await context.bot.delete_message(chat_id, user_state.last_message_id)
        except Exception:
            pass
        finally:
            user_state.last_message_id = None

    new_msg = await _send_photo_with_fallback(
        context=context,
        chat_id=chat_id,
        poster_url=poster_url,
        caption=caption,
        reply_markup=reply_markup,
        alt_poster_url=alt_poster_url,
    )

    if new_msg:
        user_state.last_message_id = new_msg.message_id
        user_state.add_shown_recommendation(film_id)


async def handle_navigation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    user_id = update.effective_user.id
    action = query.data
    user_state = user_states.get(user_id)

    if not user_state or user_state.is_blocked:
        return
    user_state.is_blocked = True

    if user_state.last_feedback_message_id:
        try:
            await context.bot.delete_message(user_id, user_state.last_feedback_message_id)
        except Exception:
            pass
        finally:
            user_state.last_feedback_message_id = None

    try:
        if action == "next":
            target = -1
            if user_state.history_index < len(user_state.shown_history) - 1:
                user_state.history_index += 1
                target = user_state.shown_history[user_state.history_index]
            elif user_state.current_index < len(user_state.recommendations) - 1:
                target = user_state.current_index + 1

            if target != -1 and 0 <= target < len(user_state.recommendations):
                user_state.current_index = target
                await send_recommendation(update, context)
            else:
                msg = await context.bot.send_message(
                    user_id, "Это была последняя рекомендация по вашему запросу."
                )
                user_state.last_feedback_message_id = msg.message_id

        elif action == "prev":
            if user_state.history_index > 0:
                user_state.history_index -= 1
                user_state.current_index = user_state.shown_history[user_state.history_index]
                await send_recommendation(update, context)
            else:
                msg = await context.bot.send_message(
                    user_id, "Это первая рекомендация в истории просмотра."
                )
                user_state.last_feedback_message_id = msg.message_id

        elif action == "like":
            msg = await context.bot.send_message(user_id, "Отзыв учтён, спасибо.")
            user_state.last_feedback_message_id = msg.message_id

        elif action == "dislike":
            msg = await context.bot.send_message(user_id, "Понял, учту в следующих рекомендациях.")
            user_state.last_feedback_message_id = msg.message_id

    except Exception as e:
        logger.error(f"Navigation error: {e}", exc_info=True)
    finally:
        user_state.is_blocked = False


async def bookmark_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    user_id = update.effective_user.id
    user_state = user_states.get(user_id)

    parts = query.data.split("_")
    if len(parts) < 2:
        await query.answer()
        return

    film_id = int(parts[1])

    current_film = None
    if user_state and user_state.recommendations:
        for f in user_state.recommendations:
            if f.get("kinopoiskId") == film_id:
                current_film = f
                break

    if not current_film or not current_film.get("nameRu"):
        async with aiohttp.ClientSession() as session:
            current_film = await recommendation_engine.get_film_details(film_id, session)

    if not current_film:
        await query.answer("Не удалось получить данные о фильме")
        return

    is_saved = db.is_in_watchlist(user_id, film_id)
    if is_saved:
        db.remove_from_watchlist(user_id, film_id)
        # Silent answer: button updates without intrusive banner
        await query.answer()
        new_icon = "В закладки"
    else:
        db.add_to_watchlist(user_id, current_film)
        await query.answer()
        new_icon = "✓ В закладках"

    if query.message and query.message.reply_markup:
        new_keyboard = []
        for row in query.message.reply_markup.inline_keyboard:
            new_row = []
            for btn in row:
                if btn.callback_data == query.data:
                    new_row.append(InlineKeyboardButton(new_icon, callback_data=btn.callback_data))
                else:
                    new_row.append(btn)
            new_keyboard.append(new_row)
        try:
            await query.edit_message_reply_markup(reply_markup=InlineKeyboardMarkup(new_keyboard))
        except Exception:
            pass
