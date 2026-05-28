"""
agro_bot.py — Telegram bot for agricultural land analysis.
Uses python-telegram-bot v20 with conversation handlers.
Data: Open-Meteo, SoilGrids, OSM, PKK Rosreestr.
AI: Groq llama-3.3-70b-versatile.
"""

import os
import json
import asyncio
import logging
import re
from typing import Dict, List, Optional

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    filters,
    ContextTypes,
    ConversationHandler,
)

from collector import collect_multiple_fields
from analyst import generate_field_report, generate_region_summary, generate_conclusion

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BOT_TOKEN: str = os.environ.get("AGRO_BOT_TOKEN", "")
GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conversation states
# ---------------------------------------------------------------------------

MAIN_MENU = 0
COLLECTING_COORDS = 1
CONFIRMING = 2
SELECTING_PERIOD = 3

# ---------------------------------------------------------------------------
# Per-user session storage
# ---------------------------------------------------------------------------

user_sessions: Dict[int, Dict] = {}


def get_session(user_id: int) -> Dict:
    """Return (and lazily create) the session dict for a user."""
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            "fields": [],
            "step": "waiting_coords",
        }
    return user_sessions[user_id]


def clear_session(user_id: int) -> None:
    """Reset the user session."""
    user_sessions[user_id] = {
        "fields": [],
        "step": "waiting_coords",
    }


# ---------------------------------------------------------------------------
# Keyboard helpers
# ---------------------------------------------------------------------------

def get_main_menu_kb() -> InlineKeyboardMarkup:
    """Return the main menu inline keyboard."""
    keyboard = [
        [InlineKeyboardButton("🌾 Анализировать поля", callback_data="menu_analyze")],
        [InlineKeyboardButton("ℹ️ Помощь", callback_data="menu_help")],
        [InlineKeyboardButton("📋 Примеры координат", callback_data="menu_examples")],
    ]
    return InlineKeyboardMarkup(keyboard)


def get_collecting_kb() -> InlineKeyboardMarkup:
    """Return the keyboard shown while collecting coordinates."""
    keyboard = [
        [InlineKeyboardButton("✅ Начать анализ", callback_data="confirm_analyze")],
        [InlineKeyboardButton("🗑️ Очистить список", callback_data="confirm_clear")],
        [InlineKeyboardButton("⬅️ Главное меню", callback_data="confirm_back")],
    ]
    return InlineKeyboardMarkup(keyboard)


def get_period_kb() -> InlineKeyboardMarkup:
    """Return the period selection keyboard."""
    keyboard = [
        [InlineKeyboardButton("📅 1 год (быстро)", callback_data="period_1")],
        [InlineKeyboardButton("📅 3 года (средняя точность)", callback_data="period_3")],
        [InlineKeyboardButton("📅 5 лет (максимум данных)", callback_data="period_5")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="period_back")],
    ]
    return InlineKeyboardMarkup(keyboard)


# ---------------------------------------------------------------------------
# Coordinate parsing
# ---------------------------------------------------------------------------

def parse_coordinates(text: str) -> List[Dict]:
    """
    Parse one or more fields from user text.

    Supported formats (one per line):
      55.3964 37.2390
      55.3964 37.2390 Участок 1
      55.3964, 37.2390
      55.3964, 37.2390 Мой участок

    Returns a list of dicts: [{"lat": float, "lon": float, "name": str}, ...]
    Silently skips lines that cannot be parsed.
    """
    fields: List[Dict] = []
    lines = text.strip().splitlines()

    coord_re = re.compile(
        r"^\s*(-?\d{1,3}(?:\.\d+)?)\s*[,\s]\s*(-?\d{1,3}(?:\.\d+)?)\s*(.*)?$"
    )

    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = coord_re.match(line)
        if not match:
            continue

        try:
            lat = float(match.group(1))
            lon = float(match.group(2))
        except ValueError:
            continue

        # Basic sanity check
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            continue

        name_part = (match.group(3) or "").strip()
        name = name_part if name_part else f"Участок ({lat}, {lon})"
        fields.append({"lat": lat, "lon": lon, "name": name})

    return fields


# ---------------------------------------------------------------------------
# Message splitting
# ---------------------------------------------------------------------------

def split_message(text: str, max_len: int = 4000) -> List[str]:
    """
    Split text into chunks at paragraph breaks so each chunk is <= max_len chars.
    Falls back to splitting at newlines, then at spaces.
    """
    if len(text) <= max_len:
        return [text]

    chunks: List[str] = []
    remaining = text

    while len(remaining) > max_len:
        # Try to split at a double newline (paragraph break)
        split_at = remaining.rfind("\n\n", 0, max_len)
        if split_at == -1:
            # Fall back to single newline
            split_at = remaining.rfind("\n", 0, max_len)
        if split_at == -1:
            # Fall back to last space
            split_at = remaining.rfind(" ", 0, max_len)
        if split_at == -1:
            # Hard cut
            split_at = max_len

        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()

    if remaining:
        chunks.append(remaining)

    return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# Helper: safe send (handles message too long etc.)
# ---------------------------------------------------------------------------

async def safe_send(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    text: str,
    reply_markup=None,
    parse_mode: Optional[str] = None,
) -> None:
    """Send a message, splitting it into chunks if it exceeds Telegram's limit."""
    chunks = split_message(text, max_len=4000)
    for i, chunk in enumerate(chunks):
        kb = reply_markup if i == len(chunks) - 1 else None
        try:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=chunk,
                reply_markup=kb,
                parse_mode=parse_mode,
            )
        except Exception as exc:
            logger.error("Failed to send message chunk: %s", exc)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle /start — show welcome message and main menu."""
    user = update.effective_user
    clear_session(user.id)

    welcome = (
        f"Привет, {user.first_name}! 👋\n\n"
        "Я — ZemleustroBot, агрономический помощник для анализа земельных участков.\n\n"
        "🌱 Что я умею:\n"
        "• Анализировать климат, почву и рельеф по координатам\n"
        "• Оценивать инфраструктуру: дороги, ЛЭП, водотоки\n"
        "• Проверять кадастровые данные (ПКК Росреестр)\n"
        "• Составлять профессиональные агрономические отчёты\n\n"
        "Используются только бесплатные открытые источники данных.\n\n"
        "Выберите действие:"
    )
    await safe_send(update, context, welcome, reply_markup=get_main_menu_kb())
    return MAIN_MENU


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help — show detailed help."""
    help_text = (
        "📖 СПРАВКА ПО ZEMLEUSTROBOT\n\n"
        "Бот анализирует сельскохозяйственные земельные участки по GPS-координатам.\n\n"
        "ИСТОЧНИКИ ДАННЫХ:\n"
        "• Климат — Open-Meteo Archive (ERA5, ретроспектива)\n"
        "• Рельеф — SRTM через Open-Meteo Elevation\n"
        "• Почва — SoilGrids v2.0 (ISRIC)\n"
        "• Инфраструктура — OpenStreetMap (Overpass API)\n"
        "• Кадастр — ПКК Росреестр\n"
        "• ИИ-агроном — Groq llama-3.3-70b\n\n"
        "КАК ИСПОЛЬЗОВАТЬ:\n"
        "1. Нажмите «🌾 Анализировать поля»\n"
        "2. Введите координаты участка (широта пробел долгота)\n"
        "3. При желании добавьте ещё до 5 участков\n"
        "4. Нажмите «✅ Начать анализ»\n"
        "5. Дождитесь отчёта (обычно 1–3 минуты)\n\n"
        "ФОРМАТЫ КООРДИНАТ:\n"
        "55.3964 37.2390\n"
        "55.3964 37.2390 Название участка\n"
        "55.3964, 37.2390\n\n"
        "КОМАНДЫ:\n"
        "/start — главное меню\n"
        "/analyze — начать анализ\n"
        "/help — эта справка\n\n"
        "⚠️ Данные ПКК Росреестр доступны только для участков в России."
    )
    await safe_send(update, context, help_text)


async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle /analyze — jump to coordinate collection."""
    user = update.effective_user
    clear_session(user.id)

    prompt_text = (
        "📍 Введите координаты поля в формате:\n\n"
        "<code>55.3964 37.2390</code>\n"
        "или с именем:\n"
        "<code>55.3964 37.2390 Участок №1</code>\n\n"
        "Можно добавить несколько полей (до 5), каждое с новой строки.\n"
        "Нажмите «✅ Начать анализ» когда добавите все поля."
    )
    await safe_send(update, context, prompt_text, parse_mode="HTML")
    return COLLECTING_COORDS


# ---------------------------------------------------------------------------
# Callback handlers (inline buttons)
# ---------------------------------------------------------------------------

async def callback_menu_analyze(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """User clicked 'Анализировать поля'."""
    query = update.callback_query
    await query.answer()

    user = update.effective_user
    clear_session(user.id)

    prompt_text = (
        "📍 Введите координаты поля в формате:\n\n"
        "55.3964 37.2390\n"
        "или с именем:\n"
        "55.3964 37.2390 Участок №1\n\n"
        "Можно добавить несколько полей (до 5), каждое с новой строки.\n"
        "Нажмите «✅ Начать анализ» когда добавите все поля."
    )
    await query.edit_message_text(prompt_text)
    return COLLECTING_COORDS


async def callback_menu_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """User clicked 'Помощь'."""
    query = update.callback_query
    await query.answer()

    help_text = (
        "📖 СПРАВКА\n\n"
        "Бот анализирует сельскохозяйственные участки по координатам.\n\n"
        "ДАННЫЕ: климат (ERA5), рельеф (SRTM), почва (SoilGrids), "
        "инфраструктура (OSM), кадастр (Росреестр)\n\n"
        "ФОРМАТЫ:\n"
        "55.3964 37.2390\n"
        "55.3964 37.2390 Название\n"
        "55.3964, 37.2390\n\n"
        "Нажмите «🌾 Анализировать поля» чтобы начать."
    )
    await query.edit_message_text(help_text, reply_markup=get_main_menu_kb())
    return MAIN_MENU


async def callback_menu_examples(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """User clicked 'Примеры координат'."""
    query = update.callback_query
    await query.answer()

    examples_text = (
        "📋 ПРИМЕРЫ КООРДИНАТ ДЛЯ РАЗНЫХ РЕГИОНОВ РОССИИ\n\n"
        "Краснодарский край (Кубань):\n"
        "45.0360 38.9760 Кубанское поле\n\n"
        "Ростовская область:\n"
        "47.2226 39.7186 Поле Дон\n\n"
        "Ставропольский край:\n"
        "45.0448 41.9734 Ставрополье\n\n"
        "Московская область:\n"
        "55.7558 37.6176 Подмосковье\n\n"
        "Воронежская область:\n"
        "51.6755 39.2088 Черноземье\n\n"
        "Западная Сибирь (Алтайский край):\n"
        "53.3479 83.7798 Алтай\n\n"
        "Поволжье (Саратовская обл.):\n"
        "51.5723 46.0350 Поволжье\n\n"
        "Скопируйте нужные координаты и отправьте их боту через «🌾 Анализировать поля»."
    )
    await query.edit_message_text(examples_text, reply_markup=get_main_menu_kb())
    return MAIN_MENU


async def callback_confirm_analyze(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """User clicked 'Начать анализ' — show period selection."""
    query = update.callback_query
    await query.answer()
    user = update.effective_user
    session = get_session(user.id)

    fields = session.get("fields", [])
    if not fields:
        await query.edit_message_text(
            "⚠️ Список участков пуст. Введите координаты хотя бы одного участка.",
            reply_markup=get_collecting_kb(),
        )
        return COLLECTING_COORDS

    if len(fields) > 5:
        session["fields"] = fields[:5]

    n = len(session["fields"])
    names = "\n".join(f"  {i+1}. {f['name']}" for i, f in enumerate(session["fields"]))
    await query.edit_message_text(
        f"📋 Участков: {n}\n{names}\n\n"
        "📅 За какой период анализировать климатические данные?",
        reply_markup=get_period_kb(),
    )
    return SELECTING_PERIOD


async def _run_analysis(
    update: Update, context: ContextTypes.DEFAULT_TYPE, years: int
) -> int:
    """Run data collection and report generation for the current session."""
    user = update.effective_user
    session = get_session(user.id)
    fields = session.get("fields", [])
    chat_id = update.effective_chat.id

    session["step"] = "analyzing"

    await context.bot.send_message(
        chat_id=chat_id,
        text=(
            f"⏳ Собираю данные для {len(fields)} участка(ов) за {years} год(а/лет)...\n"
            "Это может занять 1–3 минуты. Пожалуйста, подождите."
        ),
    )

    try:
        all_data = await collect_multiple_fields(fields, years=years)
    except Exception as exc:
        logger.exception("collect_multiple_fields failed")
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"❌ Ошибка при сборе данных: {exc}\n\nПопробуйте ещё раз.",
            reply_markup=get_main_menu_kb(),
        )
        clear_session(user.id)
        return MAIN_MENU

    await context.bot.send_message(
        chat_id=chat_id,
        text="📊 Данные собраны. Генерирую агрономический анализ...",
    )

    for field_id, field_data in all_data.items():
        field_name = field_data.get("meta", {}).get("name", field_id)
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"📝 Составляю отчёт для: {field_name}...",
        )
        try:
            report = generate_field_report(field_data, GROQ_API_KEY)
        except Exception as exc:
            logger.exception("generate_field_report failed for %s", field_id)
            report = f"Ошибка при генерации отчёта: {exc}"

        header = f"🌾 ОТЧЁТ: {field_name}\n{'═' * 40}\n\n"
        for chunk in split_message(header + report, max_len=4000):
            try:
                await context.bot.send_message(chat_id=chat_id, text=chunk)
            except Exception as exc:
                logger.error("Failed to send report chunk: %s", exc)

    if len(all_data) > 1:
        await context.bot.send_message(
            chat_id=chat_id,
            text="🔍 Составляю сравнительный анализ участков...",
        )
        try:
            conclusion = generate_conclusion(all_data, GROQ_API_KEY)
        except Exception as exc:
            logger.exception("generate_conclusion failed")
            conclusion = f"Ошибка при генерации сравнительного анализа: {exc}"

        header = f"📊 СРАВНИТЕЛЬНЫЙ АНАЛИЗ\n{'═' * 40}\n\n"
        for chunk in split_message(header + conclusion, max_len=4000):
            try:
                await context.bot.send_message(chat_id=chat_id, text=chunk)
            except Exception as exc:
                logger.error("Failed to send conclusion chunk: %s", exc)

    await context.bot.send_message(
        chat_id=chat_id,
        text="✅ Анализ завершён! Выберите следующее действие:",
        reply_markup=get_main_menu_kb(),
    )
    clear_session(user.id)
    return MAIN_MENU


async def callback_period_selected(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """User selected an analysis period (1/3/5 years) or pressed Back."""
    query = update.callback_query
    await query.answer()
    data = query.data or ""

    if data == "period_back":
        session = get_session(update.effective_user.id)
        fields = session.get("fields", [])
        lines = ["📋 Текущий список участков:"]
        for i, f in enumerate(fields, 1):
            lines.append(f"  {i}. {f['name']} ({f['lat']}, {f['lon']})")
        lines.append("\nДобавьте ещё участки или нажмите «✅ Начать анализ».")
        await query.edit_message_text("\n".join(lines), reply_markup=get_collecting_kb())
        return COLLECTING_COORDS

    period_map = {"period_1": 1, "period_3": 3, "period_5": 5}
    years = period_map.get(data)
    if years is None:
        await query.answer("Неизвестный период")
        return SELECTING_PERIOD

    await query.edit_message_text(f"⏳ Выбран период: {years} год(а/лет). Начинаю сбор данных...")
    return await _run_analysis(update, context, years)


async def callback_confirm_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """User clicked 'Очистить список'."""
    query = update.callback_query
    await query.answer()
    user = update.effective_user
    clear_session(user.id)

    await query.edit_message_text(
        "🗑️ Список участков очищен.\n\n"
        "Введите координаты нового участка:",
    )
    return COLLECTING_COORDS


async def callback_confirm_back(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """User clicked 'Главное меню'."""
    query = update.callback_query
    await query.answer()
    user = update.effective_user
    clear_session(user.id)

    await query.edit_message_text(
        "Главное меню. Выберите действие:",
        reply_markup=get_main_menu_kb(),
    )
    return MAIN_MENU


# ---------------------------------------------------------------------------
# Message handler for coordinate collection
# ---------------------------------------------------------------------------

async def coords_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle text messages while in COLLECTING_COORDS state."""
    user = update.effective_user
    session = get_session(user.id)
    text = update.message.text or ""

    parsed = parse_coordinates(text)

    if not parsed:
        await update.message.reply_text(
            "⚠️ Не удалось распознать координаты.\n\n"
            "Используйте один из форматов:\n"
            "55.3964 37.2390\n"
            "55.3964 37.2390 Название участка\n"
            "55.3964, 37.2390\n\n"
            "Широта и долгота разделяются пробелом или запятой.\n"
            "Можно указать несколько участков — каждый с новой строки.",
            reply_markup=get_collecting_kb() if session.get("fields") else None,
        )
        return COLLECTING_COORDS

    # Check capacity limit
    current_count = len(session.get("fields", []))
    remaining = 5 - current_count
    if remaining <= 0:
        await update.message.reply_text(
            "⚠️ Достигнут лимит в 5 участков.\n"
            "Нажмите «✅ Начать анализ» или «🗑️ Очистить список».",
            reply_markup=get_collecting_kb(),
        )
        return COLLECTING_COORDS

    added = parsed[:remaining]
    session["fields"].extend(added)

    skipped = len(parsed) - len(added)

    # Build confirmation text
    lines = [f"✅ Добавлено участков: {len(added)}"]
    if skipped > 0:
        lines.append(f"(пропущено {skipped} — превышен лимит 5 участков)")
    lines.append("")
    lines.append("📋 Текущий список участков:")
    for i, f in enumerate(session["fields"], 1):
        lines.append(f"  {i}. {f['name']} ({f['lat']}, {f['lon']})")
    lines.append("")
    lines.append("Добавьте ещё участки или нажмите «✅ Начать анализ».")

    await update.message.reply_text("\n".join(lines), reply_markup=get_collecting_kb())
    return COLLECTING_COORDS


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point — build and run the bot."""
    if not BOT_TOKEN:
        raise ValueError(
            "AGRO_BOT_TOKEN не задан. "
            "Установите переменную окружения AGRO_BOT_TOKEN."
        )

    app = Application.builder().token(BOT_TOKEN).build()

    # Callback routing table
    main_menu_callbacks = {
        "menu_analyze": callback_menu_analyze,
        "menu_help": callback_menu_help,
        "menu_examples": callback_menu_examples,
    }

    collecting_callbacks = {
        "confirm_analyze": callback_confirm_analyze,
        "confirm_clear": callback_confirm_clear,
        "confirm_back": callback_confirm_back,
        "menu_analyze": callback_menu_analyze,
        "menu_help": callback_menu_help,
        "menu_examples": callback_menu_examples,
    }

    async def main_menu_callback_router(
        update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        query = update.callback_query
        data = query.data or ""
        handler = main_menu_callbacks.get(data)
        if handler:
            return await handler(update, context)
        await query.answer("Неизвестная команда")
        return MAIN_MENU

    async def collecting_callback_router(
        update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        query = update.callback_query
        data = query.data or ""
        handler = collecting_callbacks.get(data)
        if handler:
            return await handler(update, context)
        await query.answer("Неизвестная команда")
        return COLLECTING_COORDS

    conv = ConversationHandler(
        entry_points=[
            CommandHandler("start", start_handler),
            CommandHandler("analyze", analyze_command),
        ],
        states={
            MAIN_MENU: [
                CallbackQueryHandler(main_menu_callback_router),
            ],
            COLLECTING_COORDS: [
                MessageHandler(
                    filters.TEXT & ~filters.COMMAND, coords_message_handler
                ),
                CallbackQueryHandler(collecting_callback_router),
            ],
            CONFIRMING: [
                CallbackQueryHandler(collecting_callback_router),
            ],
            SELECTING_PERIOD: [
                CallbackQueryHandler(callback_period_selected),
            ],
        },
        fallbacks=[
            CommandHandler("start", start_handler),
        ],
        allow_reentry=True,
    )

    app.add_handler(conv)
    app.add_handler(CommandHandler("help", help_command))

    # Deployment: webhook on Render, polling locally
    print("🤖 Запуск в режиме polling...")
    app.run_polling()


if __name__ == "__main__":
    main()
