import os
import shutil
import uuid
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Header, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from sqlalchemy import text

load_dotenv()

from utils.semantic_utils import semantic_search
from utils.answer_generator import generate_answer
import database
import models
from ingest import process_and_ingest_pdf

# ── API Key Security ──────────────────────────────────────────────────────────
BOOKFRIEND_API_KEY = os.getenv("BOOKFRIEND_API_KEY")

def verify_api_key(x_api_key: Optional[str] = Header(None)):
    if not BOOKFRIEND_API_KEY:
        raise HTTPException(status_code=500, detail="Server misconfiguration: BOOKFRIEND_API_KEY not set.")
    if x_api_key != BOOKFRIEND_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid or missing X-API-Key header.")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="BookFriend API",
    version="3.0 (Supabase + pgvector Edition)"
)

@app.on_event("startup")
def startup_event():
    database.init_db()
    print("✅ Application startup complete. Connected to Supabase.")

# ── Request / Response Models ─────────────────────────────────────────────────
class IngestResponse(BaseModel):
    message: str
    job_id: str          # ← now returns a job_id, not a book_id directly
    status: str

class JobStatusResponse(BaseModel):
    job_id: str
    book_id: Optional[str]
    filename: str
    status: str          # pending | processing | done | failed
    error: Optional[str]

class QueryRequest(BaseModel):
    user_id: str
    book_id: str
    query: str
    chapter_limit: int

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

class BookListResponse(BaseModel):
    id: str
    title: str
    filename: str

class DeleteResponse(BaseModel):
    message: str
    book_id: str

# ── Background Worker ─────────────────────────────────────────────────────────
def _run_ingest(job_id: str, pdf_path: str, original_filename: str, safe_filename: str):
    """
    Runs in the background after the HTTP response has already been sent.
    Handles the full ingest pipeline and updates job status throughout.
    """
    try:
        database.update_job(job_id, status="processing")

        book_id = database.register_book(
            title=original_filename,
            filename=safe_filename,
            index_path="supabase-pgvector"
        )

        database.update_job(job_id, status="processing", book_id=book_id)

        process_and_ingest_pdf(pdf_path, book_id)

        database.update_job(job_id, status="done", book_id=book_id)
        print(f"✅ Job {job_id} complete — book_id: {book_id}")

    except Exception as e:
        database.update_job(job_id, status="failed", error=str(e))
        print(f"❌ Job {job_id} failed: {e}")
    finally:
        # Always clean up the temp PDF, even if processing failed
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

# ── Public Endpoints ──────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {"status": "online", "version": "3.0", "vector_db": "supabase-pgvector"}

# ── Protected Endpoints ───────────────────────────────────────────────────────
@app.get("/v1/books", response_model=List[BookListResponse], dependencies=[Depends(verify_api_key)])
def list_books(db: Session = Depends(database.get_db)):
    rows = db.execute(text("SELECT id, title, filename FROM books")).mappings().fetchall()
    return [{"id": r["id"], "title": r["title"], "filename": r["filename"]} for r in rows]


@app.post("/v1/ingest", response_model=IngestResponse, dependencies=[Depends(verify_api_key)])
def ingest_book(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    safe_filename = file.filename.replace(" ", "_")

    # ── Duplicate protection ──────────────────────────────────────────────────
    if database.book_exists_by_filename(safe_filename):
        raise HTTPException(
            status_code=409,
            detail=f"A book with filename '{safe_filename}' already exists. "
                   f"Delete it first via DELETE /v1/books/<book_id>."
        )

    # ── Save PDF to /tmp immediately (before returning) ──────────────────────
    pdf_path = f"/tmp/temp_{uuid.uuid4().hex[:8]}.pdf"
    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # ── Create job record ─────────────────────────────────────────────────────
    job_id = uuid.uuid4().hex[:12]
    database.create_job(job_id, safe_filename)

    # ── Schedule background processing — response returns NOW ─────────────────
    background_tasks.add_task(
        _run_ingest,
        job_id=job_id,
        pdf_path=pdf_path,
        original_filename=file.filename,
        safe_filename=safe_filename
    )

    print(f"📋 Job {job_id} queued for '{safe_filename}' — returning immediately.")

    # ← This returns to the Android app in < 1 second
    return {
        "message": "Book upload received. Processing in background.",
        "job_id": job_id,
        "status": "pending"
    }


@app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse, dependencies=[Depends(verify_api_key)])
def get_job_status(job_id: str):
    """
    Poll this endpoint to check if your book has finished processing.
    status: pending → processing → done (or failed)
    Once status is 'done', the book_id is available to use in /v1/query.
    """
    job = database.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job


@app.delete("/v1/books/{book_id}", response_model=DeleteResponse, dependencies=[Depends(verify_api_key)])
def delete_book(book_id: str):
    """Permanently deletes a book and ALL its data."""
    deleted = database.delete_book(book_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Book '{book_id}' not found.")
    return {
        "message": f"Book '{book_id}' and all its data has been permanently deleted.",
        "book_id": book_id
    }


@app.post("/v1/query", response_model=QueryResponse, dependencies=[Depends(verify_api_key)])
def query_book(req: QueryRequest, db: Session = Depends(database.get_db)):
    # 1. Look up the real book title
    book_row = db.execute(
        text("SELECT title FROM books WHERE id = :id"),
        {"id": req.book_id}
    ).mappings().fetchone()

    if not book_row:
        raise HTTPException(status_code=404, detail=f"Book '{req.book_id}' not found.")

    book_title = book_row["title"]

    # 2. Conversation history
    history = database.get_chat_history(req.user_id, req.book_id)

    class MemoryWrapper:
        def get_context(self, limit=6): return history

    # 3. Semantic search
    raw_results = semantic_search(
        query=req.query,
        book_id=req.book_id,
        chapter_limit=req.chapter_limit,
        top_k=3
    )

    chunks_text = [chunk for _, chunk, _ in raw_results]
    sources = [source for source, _, _ in raw_results]

    if not chunks_text:
        return {"answer": "I couldn't find anything about that in the book up to this chapter.", "sources": []}

    # 4. Generate answer
    answer = generate_answer(
        query=req.query,
        context_chunks=chunks_text,
        memory=MemoryWrapper(),
        book_title=book_title
    )

    # 5. Log to history
    database.log_message(req.user_id, req.book_id, "user", req.query, req.chapter_limit)
    database.log_message(req.user_id, req.book_id, "bot", answer, req.chapter_limit)

    return {"answer": answer, "sources": sources}