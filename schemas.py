from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime

class CompanyCreate(BaseModel):
    name: str
    slug: str
    email: EmailStr

class CompanyOut(BaseModel):
    id: int
    name: str
    slug: str
    email: EmailStr
    logo_path: Optional[str]
    qr_path: Optional[str]
    prompt: Optional[str]

    class Config:
        orm_mode = True

class ReviewOut(BaseModel):
    id: int
    company_id: int
    user_name: Optional[str]
    raw_text: Optional[str]
    normalized_text: Optional[str]
    sentiment: Optional[str]
    status: str
    created_at: datetime

    class Config:
        orm_mode = True

class ReviewFinalizeIn(BaseModel):
    text: str   # окончательный текст, после редактирования
    user_name: Optional[str] = None


class CompanyPromptUpdate(BaseModel):
    prompt: str
