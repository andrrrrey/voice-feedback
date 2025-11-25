# Voice Feedback Service (MVP)

Веб-сервис для сбора голосовых отзывов клиентов с распознаванием речи и отправкой итогового текста на email компании.

## Стек

- Python 3.10+
- FastAPI
- SQLite + SQLAlchemy
- Jinja2 шаблоны
- qrcode для генерации QR-кодов
- Яндекс SpeechKit + YandexGPT (пока заглушки в `ai_utils.py`)

## Быстрый старт (локально)

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # и заполнить SMTP / логин/пароль
uvicorn main:app --reload
```

Открой:

- http://127.0.0.1:8000/admin/login — вход в админку
- http://127.0.0.1:8000/r/{slug} — публичная форма для компании
```
