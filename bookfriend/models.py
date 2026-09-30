from bookfriend.db.database import Base
from sqlalchemy import Column, String, Integer, DateTime, Text
from datetime import datetime, timezone
from pgvector.sqlalchemy import Vector

class Book(Base):
    __tablename__ = "books"
    id = Column(String, primary_key=True, index=True)
    title = Column(String)
    filename = Column(String)
    index_path = Column(String)

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, index=True)
    book_id = Column(String, index=True)
    role = Column(String)
    content = Column(Text)
    chapter_limit = Column(Integer, nullable=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class BookChunk(Base):
    __tablename__ = "book_chunks"
    id = Column(Integer, primary_key=True, autoincrement=True)
    book_id = Column(String, index=True)
    chapter_num = Column(Integer, index=True)
    chunk_text = Column(Text)
    embedding = Column(Vector(3072))

class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class IngestJob(Base):
    __tablename__ = "ingest_jobs"
    id = Column(String, primary_key=True, index=True)
    book_id = Column(String, nullable=True)
    filename = Column(String)
    status = Column(String) # pending, processing, completed, failed
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
