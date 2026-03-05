"""
Утилиты для интеграции с Bitrix24.
При финализации отзыва создаёт новый лид в CRM компании через входящий webhook.
"""
import logging
import requests

logger = logging.getLogger(__name__)

SENTIMENT_LABELS = {
    "positive": "Позитивный",
    "negative": "Негативный",
    "neutral": "Нейтральный",
}


def create_bitrix_lead(
    webhook_url: str,
    source_label: str,
    user_name: str | None,
    review_text: str,
    sentiment: str | None,
    company_name: str,
) -> bool:
    """
    Создаёт лид в Bitrix24 через REST API (входящий webhook).

    Параметры:
        webhook_url     — URL входящего webhook, напр.
                          https://yourcompany.bitrix24.ru/rest/1/xxxxxx/
        source_label    — Метка источника, напр. VOICE_FEEDBACK_FORM
        user_name       — Имя пользователя (может быть None)
        review_text     — Финальный текст отзыва
        sentiment       — Тональность: positive / negative / neutral
        company_name    — Название компании (для заголовка лида)

    Возвращает True при успехе, False при ошибке.
    """
    if not webhook_url:
        return False

    webhook_url = webhook_url.rstrip("/") + "/"
    endpoint = f"{webhook_url}crm.lead.add.json"

    sentiment_label = SENTIMENT_LABELS.get(sentiment or "", sentiment or "")
    name_display = user_name or "Аноним"

    title = f"Отзыв от {name_display} — {company_name}"
    comments = f"Тональность: {sentiment_label}\n\n{review_text}"

    payload = {
        "fields": {
            "TITLE": title,
            "NAME": name_display,
            "COMMENTS": comments,
            "SOURCE_ID": "WEB",
            "SOURCE_DESCRIPTION": source_label or "VOICE_FEEDBACK_FORM",
            "STATUS_ID": "NEW",
        }
    }

    try:
        response = requests.post(endpoint, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        if result.get("result"):
            logger.info(
                "Bitrix24 лид создан: id=%s для компании %s",
                result["result"],
                company_name,
            )
            return True
        error = result.get("error_description") or result.get("error") or str(result)
        logger.warning("Bitrix24 вернул ошибку: %s", error)
        return False
    except requests.RequestException as exc:
        logger.error("Ошибка при обращении к Bitrix24: %s", exc)
        return False
