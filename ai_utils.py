import json
import logging
import os
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
YANDEX_GPT_URL = os.getenv(
    "YANDEX_GPT_URL",
    "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
)
YANDEX_GPT_MODEL = os.getenv("YANDEX_GPT_MODEL", "yandexgpt-lite")

# Промпты можно настраивать через переменные окружения
NORMALIZATION_PROMPT = os.getenv(
    "YANDEX_GPT_NORMALIZATION_PROMPT",
    "Нормализуй текст отзыва: убери междометия, исправь опечатки и пунктуацию,"
    " сохрани смысл. Верни только нормализованный текст.",
)
SENTIMENT_PROMPT = os.getenv(
    "YANDEX_GPT_SENTIMENT_PROMPT",
    "Определи тональность текста как positive, neutral или negative, опираясь на"
    " общий эмоциональный окрас и оценочные суждения.",
)

logger = logging.getLogger(__name__)


def _auth_headers() -> dict:
    if not YANDEX_API_KEY:
        raise RuntimeError("YANDEX_API_KEY не задан")
    return {"Authorization": f"Api-Key {YANDEX_API_KEY}"}


def transcribe_audio_with_speechkit(audio_path: str) -> str:
    """
    Отправляет аудиофайл в Yandex SpeechKit и возвращает распознанный текст.
    """

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    headers = _auth_headers()
    if not YANDEX_FOLDER_ID:
        raise RuntimeError("YANDEX_FOLDER_ID не задан")

    params = {"folderId": YANDEX_FOLDER_ID, "lang": YANDEX_SPEECHKIT_LANG}

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
        logger.error("SpeechKit error: %s", response.text)
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
    return (
        f"{NORMALIZATION_PROMPT}\n\n"
        f"{SENTIMENT_PROMPT}\n"
        "Ответь строго в формате JSON: "
        '{"normalized_text": "...", "sentiment": "positive|neutral|negative"}.\n'
        "Не добавляй комментарии и пояснения."
    )


def _parse_gpt_response(text: str, fallback: str) -> Tuple[str, str]:
    try:
        data = json.loads(text)
        normalized = (data.get("normalized_text") or fallback).strip()
        sentiment = (data.get("sentiment") or "neutral").strip()
        return normalized, sentiment
    except json.JSONDecodeError:
        logger.warning("Не удалось разобрать ответ YandexGPT, возвращаем заглушку")
        return fallback, "neutral"


def normalize_and_analyze_with_yandex_gpt(raw_text: str) -> Tuple[str, str]:
    """
    Отправляет текст в YandexGPT для нормализации и определения тональности.
    Возвращает (normalized_text, sentiment).
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
            "temperature": 0.2,
            "maxTokens": 800,
        },
        "messages": [
            {
                "role": "system",
                "text": _build_gpt_prompt(),
            },
            {"role": "user", "text": raw_text},
        ],
    }

    response = requests.post(YANDEX_GPT_URL, headers=headers, json=body, timeout=30)
    if response.status_code != 200:
        logger.error("YandexGPT error: %s", response.text)
        return raw_text.strip(), "neutral"

    payload = response.json()
    try:
        text = payload["result"]["alternatives"][0]["message"]["text"]
    except (KeyError, IndexError):
        logger.warning("Неожиданный формат ответа YandexGPT: %s", payload)
        return raw_text.strip(), "neutral"

    return _parse_gpt_response(text, raw_text.strip())
