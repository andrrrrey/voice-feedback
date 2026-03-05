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
    yandex_url: Optional[str]
    twogis_url: Optional[str]
    ozon_url: Optional[str]
    wildberries_url: Optional[str]
    bitrix_webhook_url: Optional[str]
    bitrix_source_label: Optional[str]
    max_chat_id: Optional[str]

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


class CompanyLinksUpdate(BaseModel):
    yandex_url: Optional[str] = None
    twogis_url: Optional[str] = None
    ozon_url: Optional[str] = None
    wildberries_url: Optional[str] = None


class CompanyBitrixUpdate(BaseModel):
    bitrix_webhook_url: Optional[str] = None
    bitrix_source_label: Optional[str] = None


class CompanyMaxUpdate(BaseModel):
    max_chat_id: Optional[str] = None
