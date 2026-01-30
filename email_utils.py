import smtplib
import logging
from email.mime.text import MIMEText
from email.header import Header
from email.utils import formatdate, make_msgid
from typing import Optional
import time
import os
from dotenv import load_dotenv

load_dotenv()

# Настройка логирования
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("email_utils")
logger.setLevel(logging.DEBUG)

# Форматтер для логов
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Хендлер для записи в файл
file_handler = logging.FileHandler(
    os.path.join(LOG_DIR, "email.log"),
    encoding="utf-8"
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

# Хендлер для вывода в консоль
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Добавляем хендлеры только если их ещё нет
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

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
    """
    Отправляет email с отзывом на указанный адрес.

    Returns:
        True если письмо успешно отправлено, False в случае ошибки.
    """
    subject = f"Новый отзыв для компании {company_name}"
    user_part = f"Имя: {user_name}\n" if user_name else ""
    body = (
        f"{user_part}"
        f"Тональность: {sentiment}\n\n"
        f"Текст отзыва:\n{text}"
    )

    # Формируем письмо с обязательными заголовками
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = Header(subject, "utf-8")
    msg["From"] = SMTP_FROM
    msg["To"] = to_email
    msg["Reply-To"] = SMTP_FROM
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid(domain=SMTP_HOST.replace("smtp.", ""))
    msg["MIME-Version"] = "1.0"

    logger.info(f"Отправка письма: to={to_email}, company={company_name}, user={user_name or 'Аноним'}")

    for attempt in range(retries):
        try:
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as server:
                server.login(SMTP_USERNAME, SMTP_PASSWORD)
                server.sendmail(SMTP_FROM, [to_email], msg.as_string())
            logger.info(f"Письмо успешно отправлено: to={to_email}, message_id={msg['Message-ID']}")
            return True
        except Exception as e:
            logger.warning(f"Ошибка отправки (попытка {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(delay)

    logger.error(f"Не удалось отправить письмо после {retries} попыток: to={to_email}")
    return False
