import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv
import uuid
from sqlalchemy import text

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("CRITICAL: DATABASE_URL is not set.")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    # 1. Enable the pgvector extension first
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    # 2. Build the tables (books, messages, book_chunks)
    Base.metadata.create_all(bind=engine)


def register_book(title: str, filename: str, index_path: str) -> str:
    """Saves a new book entry to Supabase and returns the generated ID."""
    db = SessionLocal()
    book_id = uuid.uuid4().hex[:8]
    try:
        query = text("""
            INSERT INTO books (id, title, filename, index_path) 
            VALUES (:id, :title, :filename, :index_path)
        """)
        db.execute(query, {"id": book_id, "title": title, "filename": filename, "index_path": index_path})
        db.commit()
        return book_id
    finally:
        db.close()


def book_exists_by_filename(filename: str) -> bool:
    """Returns True if a book with this filename has already been ingested."""
    db = SessionLocal()
    try:
        row = db.execute(
            text("SELECT id FROM books WHERE filename = :filename LIMIT 1"),
            {"filename": filename}
        ).fetchone()
        return row is not None
    finally:
        db.close()


def delete_book(book_id: str) -> bool:
    """
    Deletes a book and ALL its associated data from Supabase.
    Cleans up: book_chunks (vectors) → messages (history) → books (record).
    Returns True if a book was deleted, False if the book_id didn't exist.
    """
    db = SessionLocal()
    try:
        # First check the book actually exists
        row = db.execute(
            text("SELECT id FROM books WHERE id = :id"),
            {"id": book_id}
        ).fetchone()

        if not row:
            return False

        # 1. Delete all vector chunks for this book
        db.execute(
            text("DELETE FROM book_chunks WHERE book_id = :id"),
            {"id": book_id}
        )

        # 2. Delete all chat history for this book
        db.execute(
            text("DELETE FROM messages WHERE book_id = :id"),
            {"id": book_id}
        )

        # 3. Delete the book record itself
        db.execute(
            text("DELETE FROM books WHERE id = :id"),
            {"id": book_id}
        )

        db.commit()
        return True
    except Exception as e:
        db.rollback()
        print(f"❌ Error deleting book {book_id}: {e}")
        raise
    finally:
        db.close()


def log_message(user_id: str, book_id: str, role: str, content: str, chapter_limit: int):
    """Saves a chat message to Supabase."""
    db = SessionLocal()
    try:
        query = text("""
            INSERT INTO messages (user_id, book_id, role, content, chapter_limit) 
            VALUES (:uid, :bid, :role, :content, :limit)
        """)
        db.execute(query, {
            "uid": user_id,
            "bid": book_id,
            "role": role,
            "content": content,
            "limit": chapter_limit
        })
        db.commit()
    except Exception as e:
        print(f"Error logging message: {e}")
        db.rollback()
    finally:
        db.close()


def get_chat_history(user_id: str, book_id: str):
    """Retrieves previous messages for context."""
    db = SessionLocal()
    try:
        query = text("""
            SELECT role, content FROM messages 
            WHERE user_id = :uid AND book_id = :bid 
            ORDER BY id ASC
        """)
        rows = db.execute(query, {"uid": user_id, "bid": book_id}).mappings().fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in rows]
    except Exception as e:
        print(f"Error fetching history: {e}")
        return []
    finally:
        db.close()