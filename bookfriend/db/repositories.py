import uuid

from sqlalchemy import text

from .database import SessionLocal


# ------------------------------------------------------------------
# BOOKS
# ------------------------------------------------------------------

def register_book(title: str, filename: str, index_path: str) -> str:
    db = SessionLocal()
    book_id = uuid.uuid4().hex[:8]

    try:
        db.execute(
            text("""
                INSERT INTO books
                (id, title, filename, index_path)
                VALUES
                (:id, :title, :filename, :index_path)
            """),
            {
                "id": book_id,
                "title": title,
                "filename": filename,
                "index_path": index_path,
            },
        )

        db.commit()
        return book_id

    finally:
        db.close()


def book_exists_by_filename(filename: str) -> bool:
    db = SessionLocal()

    try:
        row = db.execute(
            text("""
                SELECT id
                FROM books
                WHERE filename=:filename
                LIMIT 1
            """),
            {"filename": filename},
        ).fetchone()

        return row is not None

    finally:
        db.close()


def delete_book(book_id: str) -> bool:
    db = SessionLocal()

    try:
        row = db.execute(
            text("SELECT id FROM books WHERE id=:id"),
            {"id": book_id},
        ).fetchone()

        if not row:
            return False

        db.execute(
            text("DELETE FROM book_chunks WHERE book_id=:id"),
            {"id": book_id},
        )

        db.execute(
            text("DELETE FROM messages WHERE book_id=:id"),
            {"id": book_id},
        )

        db.execute(
            text("DELETE FROM books WHERE id=:id"),
            {"id": book_id},
        )

        db.commit()

        return True

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


# ------------------------------------------------------------------
# USERS
# ------------------------------------------------------------------

def create_user() -> str:
    db = SessionLocal()

    user_id = uuid.uuid4().hex[:16]

    try:
        db.execute(
            text("""
                INSERT INTO users(id)
                VALUES(:id)
            """),
            {"id": user_id},
        )

        db.commit()

        return user_id

    finally:
        db.close()


def user_exists(user_id: str) -> bool:
    db = SessionLocal()

    try:
        row = db.execute(
            text("""
                SELECT id
                FROM users
                WHERE id=:id
                LIMIT 1
            """),
            {"id": user_id},
        ).fetchone()

        return row is not None

    finally:
        db.close()


# ------------------------------------------------------------------
# INGEST JOBS
# ------------------------------------------------------------------

def create_job(job_id: str, filename: str):
    db = SessionLocal()

    try:
        db.execute(
            text("""
                INSERT INTO ingest_jobs
                (id, filename, status)
                VALUES
                (:id, :filename, 'pending')
            """),
            {
                "id": job_id,
                "filename": filename,
            },
        )

        db.commit()

    finally:
        db.close()


def update_job(job_id: str,
               status: str,
               book_id: str = None,
               error: str = None):

    db = SessionLocal()

    try:
        db.execute(
            text("""
                UPDATE ingest_jobs

                SET
                    status=:status,
                    book_id=COALESCE(:book_id, book_id),
                    error=:error,
                    updated_at=NOW()

                WHERE id=:id
            """),
            {
                "id": job_id,
                "status": status,
                "book_id": book_id,
                "error": error,
            },
        )

        db.commit()

    finally:
        db.close()


def get_job(job_id: str):
    db = SessionLocal()

    try:
        row = db.execute(
            text("""
                SELECT
                    id,
                    book_id,
                    filename,
                    status,
                    error,
                    created_at,
                    updated_at

                FROM ingest_jobs

                WHERE id=:id
            """),
            {"id": job_id},
        ).mappings().fetchone()

        return dict(row) if row else None

    finally:
        db.close()


# ------------------------------------------------------------------
# CHAT HISTORY
# ------------------------------------------------------------------

def log_message(user_id,
                book_id,
                role,
                content,
                chapter_limit):

    db = SessionLocal()

    try:
        db.execute(
            text("""
                INSERT INTO messages

                (
                    user_id,
                    book_id,
                    role,
                    content,
                    chapter_limit
                )

                VALUES

                (
                    :uid,
                    :bid,
                    :role,
                    :content,
                    :limit
                )
            """),
            {
                "uid": user_id,
                "bid": book_id,
                "role": role,
                "content": content,
                "limit": chapter_limit,
            },
        )

        db.commit()

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


def get_chat_history(user_id, book_id):
    db = SessionLocal()

    try:
        rows = db.execute(
            text("""
                SELECT role, content

                FROM
                (
                    SELECT
                        role,
                        content,
                        id

                    FROM messages

                    WHERE
                        user_id=:uid
                        AND
                        book_id=:bid

                    ORDER BY id DESC

                    LIMIT 12
                ) x

                ORDER BY id ASC
            """),
            {
                "uid": user_id,
                "bid": book_id,
            },
        ).mappings().fetchall()

        return [
            {
                "role": row["role"],
                "content": row["content"],
            }
            for row in rows
        ]

    finally:
        db.close()