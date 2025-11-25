import os
from pathlib import Path
from typing import List, Optional
import csv
from io import StringIO

from fastapi import FastAPI, Depends, Request, Form, UploadFile, File, HTTPException, status
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
import qrcode

from database import Base, engine, SessionLocal
from models import Company, Review
from schemas import CompanyCreate, CompanyOut, ReviewOut, ReviewFinalizeIn
from email_utils import send_review_email
from ai_utils import transcribe_audio_with_speechkit, normalize_and_analyze_with_yandex_gpt

# Инициализация БД
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Voice Feedback Service")

# Статика и шаблоны
BASE_DIR = Path(__file__).resolve().parent
static_dir = BASE_DIR / "static"
templates_dir = BASE_DIR / "templates"

static_dir.mkdir(exist_ok=True)
(static_dir / "logos").mkdir(exist_ok=True)
(static_dir / "qr").mkdir(exist_ok=True)
(static_dir / "audio").mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(templates_dir))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


ADMIN_LOGIN = os.getenv("ADMIN_LOGIN", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")


@app.get("/admin/login", response_class=HTMLResponse)
async def admin_login_page(request: Request):
    return templates.TemplateResponse("admin_login.html", {"request": request})


@app.post("/admin/login", response_class=HTMLResponse)
async def admin_login(
    request: Request,
    login: str = Form(...),
    password: str = Form(...)
):
    if login == ADMIN_LOGIN and password == ADMIN_PASSWORD:
        response = templates.TemplateResponse(
            "admin_dashboard.html",
            {"request": request}
        )
        response.set_cookie("admin_auth", "1")
        return response
    return templates.TemplateResponse(
        "admin_login.html",
        {"request": request, "error": "Неверный логин или пароль"},
        status_code=status.HTTP_401_UNAUTHORIZED,
    )


def require_admin(request: Request):
    if request.cookies.get("admin_auth") != "1":
        raise HTTPException(status_code=401, detail="Not authorized")


@app.post("/api/admin/companies", response_model=CompanyOut)
def create_company(company: CompanyCreate, db: Session = Depends(get_db)):
    existing = db.query(Company).filter(Company.slug == company.slug).first()
    if existing:
        raise HTTPException(status_code=400, detail="Slug already in use")

    db_company = Company(
        name=company.name,
        slug=company.slug,
        email=company.email,
    )
    db.add(db_company)
    db.commit()
    db.refresh(db_company)

    public_url = f"/r/{db_company.slug}"
    qr_img = qrcode.make(public_url)
    qr_path = static_dir / "qr" / f"{db_company.slug}.png"
    qr_img.save(qr_path)
    db_company.qr_path = f"/static/qr/{db_company.slug}.png"
    db.commit()
    db.refresh(db_company)

    return db_company


@app.get("/api/admin/companies", response_model=List[CompanyOut])
def list_companies(db: Session = Depends(get_db)):
    companies = db.query(Company).all()
    return companies


@app.post("/api/admin/companies/{company_id}/logo")
async def upload_logo(
    company_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    company = db.query(Company).get(company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")

    ext = os.path.splitext(file.filename)[1]
    logo_filename = f"company_{company_id}{ext}"
    logo_path = static_dir / "logos" / logo_filename

    with open(logo_path, "wb") as f:
        f.write(await file.read())

    company.logo_path = f"/static/logos/{logo_filename}"
    db.commit()
    db.refresh(company)
    return {"status": "ok", "logo_path": company.logo_path}


@app.get("/api/admin/reviews", response_model=List[ReviewOut])
def list_reviews(
    company_id: Optional[int] = None,
    db: Session = Depends(get_db),
):
    query = db.query(Review)
    if company_id:
        query = query.filter(Review.company_id == company_id)
    return query.order_by(Review.created_at.desc()).all()


@app.get("/api/admin/reviews/export")
def export_reviews_csv(
    company_id: Optional[int] = None,
    db: Session = Depends(get_db),
):
    query = db.query(Review)
    if company_id:
        query = query.filter(Review.company_id == company_id)
    reviews = query.order_by(Review.created_at.desc()).all()

    output = StringIO()
    writer = csv.writer(output, delimiter=";")
    writer.writerow(
        ["id", "company_id", "user_name", "raw_text", "normalized_text", "sentiment", "status", "created_at"]
    )
    for r in reviews:
        writer.writerow([
            r.id,
            r.company_id,
            r.user_name or "",
            (r.raw_text or "").replace("\n", " "),
            (r.normalized_text or "").replace("\n", " "),
            r.sentiment or "",
            r.status,
            r.created_at.isoformat(),
        ])

    output.seek(0)
    headers = {
        "Content-Disposition": "attachment; filename=reviews.csv"
    }
    return StreamingResponse(
        output,
        media_type="text/csv",
        headers=headers
    )


@app.get("/r/{company_slug}", response_class=HTMLResponse)
async def public_form(request: Request, company_slug: str, db: Session = Depends(get_db)):
    company = db.query(Company).filter(Company.slug == company_slug).first()
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    return templates.TemplateResponse(
        "public_form.html",
        {"request": request, "company": company}
    )


@app.post("/api/public/{company_slug}/upload-audio")
async def upload_audio(
    company_slug: str,
    user_name: Optional[str] = Form(None),
    audio: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    company = db.query(Company).filter(Company.slug == company_slug).first()
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")

    ext = os.path.splitext(audio.filename)[1] or ".webm"
    safe_user = user_name or "anon"
    audio_filename = f"{company.id}_{safe_user}_{os.urandom(4).hex()}{ext}"
    audio_path = static_dir / "audio" / audio_filename
    with open(audio_path, "wb") as f:
        f.write(await audio.read())

    raw_text = transcribe_audio_with_speechkit(str(audio_path))
    normalized_text, sentiment = normalize_and_analyze_with_yandex_gpt(raw_text)

    review = Review(
        company_id=company.id,
        user_name=user_name,
        audio_path=str(audio_path),
        raw_text=raw_text,
        normalized_text=normalized_text,
        sentiment=sentiment,
        status="draft",
    )
    db.add(review)
    db.commit()
    db.refresh(review)

    return {
        "review_id": review.id,
        "raw_text": raw_text,
        "normalized_text": normalized_text,
        "sentiment": sentiment,
    }


@app.post("/api/public/reviews/{review_id}/finalize")
def finalize_review(
    review_id: int,
    data: ReviewFinalizeIn,
    db: Session = Depends(get_db),
):
    review = db.query(Review).get(review_id)
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")

    company = review.company
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")

    review.normalized_text = data.text
    if data.user_name:
        review.user_name = data.user_name
    review.status = "final"
    db.commit()
    db.refresh(review)

    email_ok = send_review_email(
        to_email=company.email,
        company_name=company.name,
        user_name=review.user_name,
        text=review.normalized_text,
        sentiment=review.sentiment or "neutral",
    )

    if email_ok:
        review.status = "sent"
        db.commit()
        db.refresh(review)

    return {"status": "ok", "email_sent": email_ok}
