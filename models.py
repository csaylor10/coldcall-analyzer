from sqlalchemy import Column, Integer, JSON, String, Boolean, DateTime
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
import uuid

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    email = Column(String, nullable=False, unique=True, index=True)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    minutes = Column(Integer, default=10000, nullable=False)
    custom_red_flags = Column(JSON, default=[])
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))

class Job(Base):
    __tablename__ = "jobs"
    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, index=True)
    filename = Column(String)
    status = Column(String, default="pending")
    result = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    task_id = Column(String, nullable=True, index=True)