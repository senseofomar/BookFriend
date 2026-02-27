from database import Base
from sqlalchemy import Column, String, Integer, DateTime
from datetime import datetime, timezone
from pgvector.sqlalchemy import Vector

class Book(Base):
    __tablename__ = "books"
    id = Column(String, primary_key=True, index=True)
    title = Column(String)
    filename = Column(String)
    index_path = Column(String) # We will just store "supabase" here now

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, index=True)
    book_id = Column(String, index=True)
    role = Column(String)
    content = Column(String)
    chapter_limit = Column(Integer, nullable=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

# THE NEW VECTOR TABLE!
class BookChunk(Base):
    __tablename__ = "book_chunks"
    id = Column(Integer, primary_key=True, autoincrement=True)
    book_id = Column(String, index=True)
    chapter_num = Column(Integer, index=True)
    chunk_text = Column(String)
    embedding = Column(Vector(384)) # 384 dimensions for all-MiniLM-L6-v2