from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base

class Company(Base):
    __tablename__ = "companies"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    slug = Column(String, unique=True, index=True)   # для URL типа /r/{slug}
    email = Column(String, nullable=False)           # куда отправлять отзывы
    logo_path = Column(String, nullable=True)
    qr_path = Column(String, nullable=True)

    reviews = relationship("Review", back_populates="company")


class Review(Base):
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, index=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    user_name = Column(String, nullable=True)
    audio_path = Column(String, nullable=True)
    raw_text = Column(Text, nullable=True)           # как распознали
    normalized_text = Column(Text, nullable=True)    # после нормализации
    sentiment = Column(String, nullable=True)        # positive/neutral/negative
    status = Column(String, default="draft")         # draft/final/sent
    created_at = Column(DateTime, default=datetime.utcnow)

    company = relationship("Company", back_populates="reviews")
