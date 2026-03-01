"""
BookFriend API — Test Suite
Tests every critical endpoint without touching real Supabase or Groq.
All external calls are mocked.
"""

import io

import patch
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# ── Environment setup (must happen before importing api) ─────────────────────
import os
os.environ.setdefault("DATABASE_URL", "postgresql://fake:fake@localhost/fake")
os.environ.setdefault("BOOKFRIEND_API_KEY", "test-secret-key")
os.environ.setdefault("GROQ_API_KEY", "fake-groq-key")

# ── Patch DB engine creation so it doesn't try to actually connect ────────────
with patch("sqlalchemy.create_engine"), \
     patch("database.init_db"), \
     patch("sentence_transformers.SentenceTransformer"):
    from api import app
    import database

client = TestClient(app)

VALID_KEY = {"X-API-Key": "test-secret-key"}
WRONG_KEY = {"X-API-Key": "wrong-key"}


# ── Helper: inject a fake DB session via FastAPI's dependency override system ─
def make_mock_db():
    """Returns a MagicMock that acts like a SQLAlchemy session."""
    return MagicMock()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Health check — public, no auth needed
# ═══════════════════════════════════════════════════════════════════════════════

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["vector_db"] == "supabase-pgvector"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Auth — wrong or missing key rejected on all protected endpoints
# ═══════════════════════════════════════════════════════════════════════════════

def test_books_rejects_wrong_api_key():
    response = client.get("/v1/books", headers=WRONG_KEY)
    assert response.status_code == 401

def test_ingest_rejects_no_api_key():
    response = client.post("/v1/ingest")
    assert response.status_code == 401

def test_query_rejects_wrong_api_key():
    response = client.post(
        "/v1/query",
        headers=WRONG_KEY,
        json={"user_id": "u1", "book_id": "b1", "query": "hello", "chapter_limit": 5}
    )
    assert response.status_code == 401


# ═══════════════════════════════════════════════════════════════════════════════
# 3. GET /v1/books — returns list
# ═══════════════════════════════════════════════════════════════════════════════

def test_list_books_returns_empty_list():
    mock_db = make_mock_db()
    mock_db.execute.return_value.mappings.return_value.fetchall.return_value = []

    # ✅ FastAPI dependency override — the correct way to inject a mock DB
    app.dependency_overrides[database.get_db] = lambda: mock_db
    response = client.get("/v1/books", headers=VALID_KEY)
    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == []

def test_list_books_returns_books():
    mock_db = make_mock_db()
    mock_db.execute.return_value.mappings.return_value.fetchall.return_value = [
        {"id": "abc123", "title": "Lord of the Mysteries", "filename": "lord_of_mysteries.pdf"}
    ]

    app.dependency_overrides[database.get_db] = lambda: mock_db
    response = client.get("/v1/books", headers=VALID_KEY)
    app.dependency_overrides.clear()

    assert response.status_code == 200
    books = response.json()
    assert len(books) == 1
    assert books[0]["id"] == "abc123"
    assert books[0]["title"] == "Lord of the Mysteries"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. POST /v1/ingest — duplicate protection
# ═══════════════════════════════════════════════════════════════════════════════

def test_ingest_rejects_duplicate_filename():
    fake_pdf = io.BytesIO(b"%PDF-1.4 fake content")

    with patch("database.book_exists_by_filename", return_value=True):
        response = client.post(
            "/v1/ingest",
            headers=VALID_KEY,
            files={"file": ("my_book.pdf", fake_pdf, "application/pdf")}
        )

    assert response.status_code == 409
    assert "already exists" in response.json()["detail"]


# ═══════════════════════════════════════════════════════════════════════════════
# 5. POST /v1/ingest — success returns job_id immediately
# ═══════════════════════════════════════════════════════════════════════════════

def test_ingest_returns_job_id_immediately():
    fake_pdf = io.BytesIO(b"%PDF-1.4 fake content")

    with patch("database.book_exists_by_filename", return_value=False), \
         patch("database.create_job"), \
         patch("api._run_ingest"):
        response = client.post(
            "/v1/ingest",
            headers=VALID_KEY,
            files={"file": ("new_book.pdf", fake_pdf, "application/pdf")}
        )

    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "pending"
    assert data["message"] == "Book upload received. Processing in background."


# ═══════════════════════════════════════════════════════════════════════════════
# 6. GET /v1/jobs/{job_id} — status polling
# ═══════════════════════════════════════════════════════════════════════════════

def test_get_job_status_pending():
    fake_job = {
        "job_id": "abc123def456", "book_id": None,
        "filename": "my_book.pdf", "status": "pending", "error": None
    }
    with patch("database.get_job", return_value=fake_job):
        response = client.get("/v1/jobs/abc123def456", headers=VALID_KEY)

    assert response.status_code == 200
    assert response.json()["status"] == "pending"
    assert response.json()["book_id"] is None

def test_get_job_status_done():
    fake_job = {
        "job_id": "abc123def456", "book_id": "bookxyz1",
        "filename": "my_book.pdf", "status": "done", "error": None
    }
    with patch("database.get_job", return_value=fake_job):
        response = client.get("/v1/jobs/abc123def456", headers=VALID_KEY)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "done"
    assert data["book_id"] == "bookxyz1"

def test_get_job_status_not_found():
    with patch("database.get_job", return_value=None):
        response = client.get("/v1/jobs/doesnotexist", headers=VALID_KEY)
    assert response.status_code == 404

def test_get_job_status_failed():
    fake_job = {
        "job_id": "abc123def456", "book_id": None,
        "filename": "bad_book.pdf", "status": "failed",
        "error": "No text could be extracted from the PDF."
    }
    with patch("database.get_job", return_value=fake_job):
        response = client.get("/v1/jobs/abc123def456", headers=VALID_KEY)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "failed"
    assert "No text" in data["error"]


# ═══════════════════════════════════════════════════════════════════════════════
# 7. DELETE /v1/books/{book_id}
# ═══════════════════════════════════════════════════════════════════════════════

def test_delete_book_not_found():
    with patch("database.delete_book", return_value=False):
        response = client.delete("/v1/books/fakeid", headers=VALID_KEY)
    assert response.status_code == 404

def test_delete_book_success():
    with patch("database.delete_book", return_value=True):
        response = client.delete("/v1/books/abc123", headers=VALID_KEY)

    assert response.status_code == 200
    data = response.json()
    assert data["book_id"] == "abc123"
    assert "permanently deleted" in data["message"]


# ═══════════════════════════════════════════════════════════════════════════════
# 8. POST /v1/query
# ═══════════════════════════════════════════════════════════════════════════════

def test_query_book_not_found():
    mock_db = make_mock_db()
    # fetchone returns None → book not found → should 404
    mock_db.execute.return_value.mappings.return_value.fetchone.return_value = None

    app.dependency_overrides[database.get_db] = lambda: mock_db
    response = client.post(
        "/v1/query",
        headers=VALID_KEY,
        json={"user_id": "u1", "book_id": "fakeid", "query": "Who is Klein?", "chapter_limit": 5}
    )
    app.dependency_overrides.clear()

    assert response.status_code == 404

def test_query_returns_answer():
    mock_db = make_mock_db()
    mock_db.execute.return_value.mappings.return_value.fetchone.return_value = {"title": "Lord of the Mysteries"}

    app.dependency_overrides[database.get_db] = lambda: mock_db

    with patch("database.get_chat_history", return_value=[]), \
         patch("api.semantic_search", return_value=[          # ← was utils.semantic_utils.semantic_search
             ("chapter_1", "Klein Moretti is the protagonist.", 0.95)
         ]), \
         patch("api.generate_answer", return_value="Klein Moretti is the main character."), \
         patch("database.log_message"):

        response = client.post(
            "/v1/query",
            headers=VALID_KEY,
            json={"user_id": "u1", "book_id": "abc123", "query": "Who is Klein?", "chapter_limit": 5}
        )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Klein Moretti is the main character."
    assert "chapter_1" in data["sources"]

def test_query_returns_fallback_when_no_chunks():
    mock_db = make_mock_db()
    mock_db.execute.return_value.mappings.return_value.fetchone.return_value = {"title": "Some Book"}

    app.dependency_overrides[database.get_db] = lambda: mock_db

    with patch("database.get_chat_history", return_value=[]), \
         patch("utils.semantic_utils.semantic_search", return_value=[]):

        response = client.post(
            "/v1/query",
            headers=VALID_KEY,
            json={"user_id": "u1", "book_id": "abc123", "query": "What happened?", "chapter_limit": 1}
        )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "couldn't find" in response.json()["answer"]