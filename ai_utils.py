from typing import Tuple

def transcribe_audio_with_speechkit(audio_path: str) -> str:
    """
    audio_path: путь к аудиофайлу на сервере.
    Возвращает распознанный текст (сырой).

    TODO: заменить заглушку на реальный запрос в Яндекс SpeechKit.
    """
    # Заглушка:
    return "это тестовый распознанный текст отзыва"

def normalize_and_analyze_with_yandex_gpt(raw_text: str) -> Tuple[str, str]:
    """
    Возвращает:
    - нормализованный текст (без лишних слов + пунктуация)
    - sentiment: 'positive' | 'neutral' | 'negative'

    TODO: заменить заглушку на реальный запрос в ЯндексGPT.
    """
    normalized = raw_text.strip().capitalize()
    sentiment = "neutral"

    lower = normalized.lower()
    if any(w in lower for w in ["классно", "понравил", "супер", "отлично"]):
        sentiment = "positive"
    elif any(w in lower for w in ["ужас", "плохо", "разочар", "не понрав"]):
        sentiment = "negative"

    return normalized, sentiment
