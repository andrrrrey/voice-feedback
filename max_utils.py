"""
Интеграция с мессенджером Max (botapi.max.ru).

Схема работы:
  1. Администратор запускает бота в Max (нажимает Start).
  2. Бот отвечает секретным ключом — chat_id пользователя в Max.
  3. Администратор вводит этот ключ в настройках компании в админке.
  4. При финализации отзыва бот отправляет его текст на этот chat_id.

Необходимые переменные окружения:
  MAX_BOT_TOKEN  — токен бота, полученный у @MaxBot в мессенджере Max.
  MAX_BOT_API_URL — (опционально) базовый URL API, по умолчанию https://botapi.max.ru
"""

import logging
import os
import threading
import time

import requests

logger = logging.getLogger(__name__)

MAX_BOT_TOKEN: str = os.getenv("MAX_BOT_TOKEN", "")
MAX_BOT_API_URL: str = os.getenv("MAX_BOT_API_URL", "https://botapi.max.ru").rstrip("/")

SENTIMENT_LABELS = {
    "positive": "Позитивный",
    "negative": "Негативный",
    "neutral": "Нейтральный",
}


# ---------------------------------------------------------------------------
# Низкоуровневые вспомогательные функции
# ---------------------------------------------------------------------------

def _api_get(method: str, params: dict | None = None) -> dict | None:
    """GET-запрос к Bot API Max."""
    if not MAX_BOT_TOKEN:
        return None
    url = f"{MAX_BOT_API_URL}/{method}"
    p = {"access_token": MAX_BOT_TOKEN}
    if params:
        p.update(params)
    try:
        r = requests.get(url, params=p, timeout=35)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        logger.error("Max API GET %s error: %s", method, exc)
        return None


def _api_post(method: str, payload: dict, extra_params: dict | None = None) -> dict | None:
    """POST-запрос к Bot API Max."""
    if not MAX_BOT_TOKEN:
        return None
    url = f"{MAX_BOT_API_URL}/{method}"
    params = {"access_token": MAX_BOT_TOKEN}
    if extra_params:
        params.update(extra_params)
    try:
        r = requests.post(url, params=params, json=payload, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        logger.error("Max API POST %s error: %s", method, exc)
        return None


# ---------------------------------------------------------------------------
# Отправка сообщений
# ---------------------------------------------------------------------------

def send_max_message(chat_id: str, text: str) -> bool:
    """
    Отправляет текстовое сообщение пользователю через бот Max.

    Параметры:
        chat_id — идентификатор пользователя в Max (секретный ключ, числовой).
        text    — текст сообщения.

    Возвращает True при успехе, False при ошибке или если токен не задан.
    """
    if not MAX_BOT_TOKEN or not chat_id:
        return False

    # Max Bot API (TamTam-совместимый): user_id передаётся в query string,
    # тело запроса содержит только текст сообщения.
    try:
        user_id = int(chat_id)
    except ValueError:
        logger.error("Max: chat_id '%s' не является числом", chat_id)
        return False

    payload = {"text": text}
    result = _api_post("messages", payload, extra_params={"user_id": user_id})
    if result is not None:
        logger.info("Max: сообщение отправлено user_id=%s", user_id)
        return True
    return False


def send_review_via_max(
    chat_id: str,
    user_name: str | None,
    review_text: str,
    sentiment: str | None,
    company_name: str,
) -> bool:
    """
    Форматирует и отправляет отзыв администратору компании через Max.

    Возвращает True при успехе, False при ошибке.
    """
    if not MAX_BOT_TOKEN or not chat_id:
        return False

    sentiment_label = SENTIMENT_LABELS.get(sentiment or "", sentiment or "")
    name_display = user_name or "Аноним"

    text = (
        f"Новый отзыв для компании «{company_name}»\n\n"
        f"Имя: {name_display}\n"
        f"Тональность: {sentiment_label}\n\n"
        f"Текст отзыва:\n{review_text}"
    )
    return send_max_message(chat_id, text)


# ---------------------------------------------------------------------------
# Обработка входящих обновлений
# ---------------------------------------------------------------------------

def _extract_chat_id_from_update(update: dict) -> str | None:
    """
    Извлекает chat_id отправителя из обновления Max Bot API.

    Max Bot API отдаёт два типа событий:
      - bot_started   (пользователь нажал Start)
      - message_created (входящее сообщение)

    В обоих случаях нужный идентификатор — user_id отправителя,
    который используется как chat_id при отправке ответа.
    """
    update_type = update.get("update_type", "")

    if update_type == "bot_started":
        user = update.get("user", {})
        uid = user.get("user_id")
        return str(uid) if uid else None

    if update_type == "message_created":
        message = update.get("message", {})
        sender = message.get("sender", {})
        uid = sender.get("user_id")
        return str(uid) if uid else None

    return None


def _process_updates(updates: list) -> None:
    """Обрабатывает входящие события от бота Max."""
    for update in updates:
        chat_id = _extract_chat_id_from_update(update)
        if not chat_id:
            continue

        reply = (
            "Привет!\n\n"
            "Ваш секретный ключ для привязки к компании в системе Voice Feedback:\n\n"
            f"{chat_id}"
        )
        send_max_message(chat_id, reply)
        logger.info(
            "Max: отправлен секретный ключ пользователю chat_id=%s (update_type=%s)",
            chat_id,
            update.get("update_type", "?"),
        )


# ---------------------------------------------------------------------------
# Long-polling фоновый поток
# ---------------------------------------------------------------------------

_polling_thread: threading.Thread | None = None
_polling_stop = threading.Event()


def _polling_loop() -> None:
    """
    Получает обновления от Max Bot API методом long polling.
    Запускается как демон-поток при старте приложения.
    """
    marker: int | None = None
    logger.info("Max polling запущен (token=%s...)", MAX_BOT_TOKEN[:6] if MAX_BOT_TOKEN else "")

    while not _polling_stop.is_set():
        params: dict = {"limit": 100}
        if marker is not None:
            params["marker"] = marker

        data = _api_get("updates", params)

        if data is None:
            # Ошибка сети — подождём и повторим
            _polling_stop.wait(timeout=5)
            continue

        updates: list = data.get("updates", [])
        new_marker = data.get("marker")

        if updates:
            _process_updates(updates)

        if new_marker is not None:
            marker = new_marker

        if not updates:
            # Нет событий — короткая пауза перед следующим запросом
            _polling_stop.wait(timeout=1)

    logger.info("Max polling остановлен")


def start_polling() -> None:
    """
    Запускает фоновый поток long polling, если задан MAX_BOT_TOKEN.
    Безопасно вызывать повторно — повторный запуск игнорируется.
    """
    global _polling_thread

    if not MAX_BOT_TOKEN:
        logger.warning("MAX_BOT_TOKEN не задан — интеграция с мессенджером Max отключена")
        return

    if _polling_thread and _polling_thread.is_alive():
        return  # уже запущен

    _polling_stop.clear()
    _polling_thread = threading.Thread(
        target=_polling_loop,
        daemon=True,
        name="max-polling",
    )
    _polling_thread.start()


def stop_polling() -> None:
    """Останавливает фоновый поток polling (при завершении приложения)."""
    _polling_stop.set()
    if _polling_thread and _polling_thread.is_alive():
        _polling_thread.join(timeout=5)
