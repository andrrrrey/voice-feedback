import json
import logging
import os
import re
from typing import Tuple

import requests
from dotenv import load_dotenv

load_dotenv()

YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")
YANDEX_SPEECHKIT_STT_URL = os.getenv(
    "YANDEX_SPEECHKIT_STT_URL",
    "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize",
)
YANDEX_SPEECHKIT_LANG = os.getenv("YANDEX_SPEECHKIT_LANG", "ru-RU")

# 🔹 формат аудио для STT (по умолчанию — oggopus)
YANDEX_SPEECHKIT_FORMAT = os.getenv("YANDEX_SPEECHKIT_FORMAT", "oggopus")

YANDEX_GPT_URL = os.getenv(
    "YANDEX_GPT_URL",
    "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
)

# ⚠️ Модель можно сменить в .env, например на yandexgpt или yandexgpt-pro
YANDEX_GPT_MODEL = os.getenv("YANDEX_GPT_MODEL", "yandexgpt-lite")

# Промпты можно настраивать через переменные окружения

# 🔹 Промпт для РЕРАЙТА/НОРМАЛИЗАЦИИ
NORMALIZATION_PROMPT = os.getenv(
    "YANDEX_GPT_NORMALIZATION_PROMPT",
    "Ты редактор клиентских отзывов.\n"
    "Твоя задача — переписать отзыв так, чтобы он стал чище и понятнее.\n"
    "Правила:\n"
    "- сохраняй факты и исходное отношение автора (недовольство/довольство);\n"
    "- убирай слова-паразиты, повторы, устные конструкции типа \"ну\", \"как бы\";\n"
    "- исправляй опечатки и грамматику;\n"
    "- расставляй знаки препинания, делай текст цельным и читаемым;\n"
    "- можно немного уточнить или расширить формулировки, но не придумывай новых фактов.\n"
)

# 🔹 Промпт для ОЦЕНКИ ТОНАЛЬНОСТИ
SENTIMENT_PROMPT = os.getenv(
    "YANDEX_GPT_SENTIMENT_PROMPT",
    "Теперь оцени общий тон отзыва и выбери одну метку:\n"
    "- positive — явная похвала, удовлетворённость, готовность рекомендовать;\n"
    "- neutral — описательный текст без явного недовольства или восторга;\n"
    "- negative — жалобы, недовольство, плохой сервис, фразы вроде "
    "\"не рекомендую\", \"больше не приду\".\n"
    "Важно: если отзыв содержит серьёзные жалобы, медленное или плохое обслуживание, "
    "угрозы не возвращаться или не рекомендовать — выбирай negative, "
    "даже если встречаются отдельные положительные слова.\n"
)

logger = logging.getLogger(__name__)


def _auth_headers() -> dict:
    if not YANDEX_API_KEY:
        raise RuntimeError("YANDEX_API_KEY не задан")
    return {"Authorization": f"Api-Key {YANDEX_API_KEY}"}


def transcribe_audio_with_speechkit(audio_path: str) -> str:
    """
    Отправляет аудиофайл в Yandex SpeechKit и возвращает распознанный текст.
    Ожидается, что на вход подаётся OGG Opus (format=oggopus).
    """

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    headers = _auth_headers()
    headers["Content-Type"] = "application/octet-stream"
    if not YANDEX_FOLDER_ID:
        raise RuntimeError("YANDEX_FOLDER_ID не задан")

    params = {
        "folderId": YANDEX_FOLDER_ID,
        "lang": YANDEX_SPEECHKIT_LANG,
        "format": YANDEX_SPEECHKIT_FORMAT,
    }

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    response = requests.post(
        YANDEX_SPEECHKIT_STT_URL,
        params=params,
        headers=headers,
        data=audio_bytes,
        timeout=30,
    )

    if response.status_code != 200:
        logger.error(
            "SpeechKit error (status=%s): %s",
            response.status_code,
            response.text,
        )
        raise RuntimeError(
            f"SpeechKit STT request failed with status {response.status_code}"
        )

    payload = response.json()
    if payload.get("error_code"):
        raise RuntimeError(
            f"SpeechKit STT error {payload.get('error_code')}:"
            f" {payload.get('error_message')}"
        )

    return (payload.get("result") or "").strip()


def _build_gpt_prompt() -> str:
    """
    Строим единый промпт: сначала правила рерайта, затем правила тональности,
    дальше — требование вернуть строгий JSON.
    """
    return (
        f"{NORMALIZATION_PROMPT}\n\n"
        f"{SENTIMENT_PROMPT}\n\n"
        "Формат ответа:\n"
        'Верни строго один JSON-объект без Markdown, без пояснений и без лишнего текста:\n'
        '{"normalized_text": "...", "sentiment": "positive|neutral|negative"}\n'
        "Никаких комментариев, вступлений и пояснений — только JSON."
    )


def _heuristic_sentiment_from_text(text: str) -> str:
    """
    Определяем тональность ТОЛЬКО по тексту, без учёта ответа модели.
    Возвращает: 'positive', 'neutral' или 'negative'.
    """
    t = (text or "").lower()

    negative_markers = [
        "не понрав",          # "не понравились услуги", "не понравилось"
        "ужасн",              # "ужасный", "ужасно"
        "отврат",             # "отвратительно"
        "кошмар",
        "плохой", "плохая", "плохие", "плохо",
        "медленн",            # "медленное обслуживание"
        "разочарован", "разочарована", "разочарование",
        "не рекоменд",        # "не рекомендую", "не рекомендовал"
        "не советую",
        "больше не приду",
        "никогда больше",
        "обман", "развод",
        "хамств",
        "наплевательск",      # "наплевательски"
    ]

    positive_markers = [
        # убрали голое "понрав", чтобы не ловить "не понравилось"
        "очень понравилось",
        "мне понравилось",
        "нам понравилось",
        "понравилось обслуживание",
        "отличн",
        "прекрасн",
        "замечательн",
        "супер",
        "классн",
        "шикарн",
        "доволен", "довольна",
        "очень доволен", "очень довольна",
        "рекомендую",
        "советую",
        "буду обращаться еще", "буду обращаться ещё",
        "буду приходить еще", "буду приходить ещё",
        "лучшая", "лучший сервис",
        "спасибо", "благодар",
        "всё понравилось", "все понравилось",
    ]

    has_neg = any(marker in t for marker in negative_markers)

    # если явный негатив есть — он важнее всего
    if has_neg:
        return "negative"

    has_pos = any(marker in t for marker in positive_markers)

    if has_pos:
        return "positive"

    return "neutral"


def _parse_gpt_response(text: str, fallback: str) -> Tuple[str, str]:
    """
    Парсим ответ модели. Ожидаем JSON, но на всякий случай:
    - выдёргиваем первую фигурную скобку {...} из текста;
    - если не получилось — возвращаем fallback + хьюристический sentiment.
    """
    raw = (text or "").strip()

    try:
        # Пытаемся вытащить JSON-объект из текста
        match = re.search(r"\{.*\}", raw, flags=re.S)
        json_str = match.group(0) if match else raw

        data = json.loads(json_str)

        normalized = (data.get("normalized_text") or fallback).strip()
        sentiment = (data.get("sentiment") or "").strip().lower()

        if sentiment not in {"positive", "neutral", "negative"}:
            sentiment = _heuristic_sentiment_from_text(normalized)

        return normalized, sentiment

    except Exception as e:
        logger.warning(
            "Не удалось разобрать ответ YandexGPT как JSON: %r, ошибка: %s",
            raw,
            e,
        )
        normalized = fallback.strip()
        sentiment = _heuristic_sentiment_from_text(normalized)
        return normalized, sentiment


def normalize_and_analyze_with_yandex_gpt(raw_text: str) -> Tuple[str, str]:
    """
    Отправляет текст в YandexGPT для нормализации (рерайта) и определения тональности.
    Возвращает (normalized_text, sentiment).

    - normalized_text: переписанный отзыв без слов-паразитов, с пунктуацией;
    - sentiment: 'positive', 'neutral' или 'negative'.
    """
    raw_text = raw_text or ""

    if not YANDEX_FOLDER_ID:
        raise RuntimeError("YANDEX_FOLDER_ID не задан")

    headers = {"Content-Type": "application/json"}
    headers.update(_auth_headers())

    model_uri = f"gpt://{YANDEX_FOLDER_ID}/{YANDEX_GPT_MODEL}"

    body = {
        "modelUri": model_uri,
        "completionOptions": {
            "stream": False,
            "temperature": 0.4,  # чуть свободнее, чтобы рерайт действительно переписывал
            "maxTokens": 800,
        },
        "messages": [
            {
                "role": "system",
                "text": _build_gpt_prompt(),
            },
            {
                "role": "user",
                "text": raw_text,
            },
        ],
        # Просим модель вернуть именно JSON-объект
        "json_object": True,
    }

    try:
        response = requests.post(
            YANDEX_GPT_URL, headers=headers, json=body, timeout=30
        )
    except requests.RequestException as e:
        logger.error("YandexGPT request error: %s", e)
        normalized = raw_text.strip()
        sentiment = _heuristic_sentiment_from_text(normalized)
        return normalized, sentiment

    if response.status_code != 200:
        logger.error("YandexGPT error (status=%s): %s", response.status_code, response.text)
        normalized = raw_text.strip()
        sentiment = _heuristic_sentiment_from_text(normalized)
        return normalized, sentiment

    payload = response.json()
    try:
        text = payload["result"]["alternatives"][0]["message"]["text"]
    except (KeyError, IndexError) as e:
        logger.warning("Неожиданный формат ответа YandexGPT: %s (ошибка: %s)", payload, e)
        normalized = raw_text.strip()
        sentiment = _heuristic_sentiment_from_text(normalized)
        return normalized, sentiment

    # Здесь уже идёт полноценный рерайт + sentiment из JSON,
    # с fallback на хьюристику, если JSON кривой.
    normalized, sentiment = _parse_gpt_response(text, raw_text.strip())
    return normalized, sentiment
