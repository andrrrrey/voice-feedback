import smtplib
from email.mime.text import MIMEText
from email.header import Header
from typing import Optional
import time
import os
from dotenv import load_dotenv

load_dotenv()

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.yandex.ru")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USERNAME or "")

def send_review_email(
    to_email: str,
    company_name: str,
    user_name: Optional[str],
    text: str,
    sentiment: str,
    retries: int = 3,
    delay: int = 3,
) -> bool:
    subject = f"Новый отзыв для компании {company_name}"
    user_part = f"Имя: {user_name}\n" if user_name else ""
    body = (
        f"{user_part}"
        f"Тональность: {sentiment}\n\n"
        f"Текст отзыва:\n{text}"
    )

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = Header(subject, "utf-8")
    msg["From"] = SMTP_FROM
    msg["To"] = to_email

    for attempt in range(retries):
        try:
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as server:
                server.login(SMTP_USERNAME, SMTP_PASSWORD)
                server.sendmail(SMTP_FROM, [to_email], msg.as_string())
            return True
        except Exception as e:
            print(f"[EMAIL] Ошибка отправки (попытка {attempt+1}): {e}")
            time.sleep(delay)
    return False
