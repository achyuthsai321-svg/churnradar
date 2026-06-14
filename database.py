"""
src/api/database.py
SQLAlchemy setup and ORM models for storing churn scores.
"""

from sqlalchemy import create_engine, Column, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///churnradar.db")

engine       = create_engine(DATABASE_URL, connect_args={"check_same_thread": False}
                             if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base         = declarative_base()


class CustomerScore(Base):
    """Stores nightly churn prediction scores for each customer."""
    __tablename__ = "customer_scores"

    id                = Column(String, primary_key=True)   # customer_id
    customer_id       = Column(String, index=True)
    churn_probability = Column(Float)
    risk_level        = Column(String)                     # HIGH / MEDIUM / LOW
    top_reasons       = Column(Text)                       # JSON string
    scored_at         = Column(DateTime, default=datetime.utcnow)
    alert_sent        = Column(String, default="No")       # Yes / No


def init_db():
    """Create all tables. Call once at startup."""
    Base.metadata.create_all(bind=engine)
    print("[DB] Tables created.")


def get_db():
    """FastAPI dependency — yields a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
