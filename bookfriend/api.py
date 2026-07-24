import os
import shutil
import uuid
import tempfile
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Header, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from sqlalchemy import text
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

load_dotenv()

from bookfriend.utils.semantic_utils import semantic_search
from bookfriend.utils.answer_generator import generate_answer
# ✅ FIXED IMPORT
from bookfriend import db as database
from bookfriend.ingest import process_and_ingest_pdf

# ── Rate Limiter ──────────────────────────────────────────────────────────────
def get_user_id_or_ip(request: Request) -> str:
    try:
        body = request._json
        if body and "user_id" in body:
            return body["user_id"]
    except Exception:
        pass
    return get_remote_address(request)

limiter = Limiter(key_func=get_user_id_or_ip)

BOOKFRIEND_API_KEY = os.getenv("BOOKFRIEND_API_KEY")

def verify_api_key(x_api_key: Optional[str] = Header(None)):
    if not BOOKFRIEND_API_KEY:
        raise HTTPException(status_code=500, detail="Server misconfiguration: BOOKFRIEND_API_KEY not set.")
    if x_api_key != BOOKFRIEND_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid or missing X-API-Key header.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    database.init_db()
    print("✅ Application startup complete. Connected to Supabase.")
    yield

app = FastAPI(
    title="BookFriend API",
    version="3.1",
    lifespan=lifespan
)

# Enable CORS for public mobile/web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- Pydantic Models ---
class IngestResponse(BaseModel):
    message: str
    job_id: str
    status: str

class JobStatusResponse(BaseModel):
    job_id: str
    book_id: Optional[str]
    filename: str
    status: str
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

class UserResponse(BaseModel):
    user_id: str
    message: str

def _run_ingest(job_id: str, pdf_path: str, original_filename: str, safe_filename: str):
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
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

@app.get("/health")
def health_check():
    return {"status": "online", "version": "3.1", "vector_db": "supabase-pgvector"}

@app.post("/v1/users", response_model=UserResponse)
def register_user():
    user_id = database.create_user()
    return {
        "user_id": user_id,
        "message": "User registered successfully. Store this user_id — it identifies you."
    }

@app.get("/v1/books", response_model=List[BookListResponse], dependencies=[Depends(verify_api_key)])
def list_books(db: Session = Depends(database.get_db)):
    rows = db.execute(text("SELECT id, title, filename FROM books")).mappings().fetchall()
    return [{"id": r["id"], "title": r["title"], "filename": r["filename"]} for r in rows]

@app.post("/v1/ingest", response_model=IngestResponse, dependencies=[Depends(verify_api_key)])
def ingest_book(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    safe_filename = file.filename.replace(" ", "_")
    if database.book_exists_by_filename(safe_filename):
        raise HTTPException(
            status_code=409,
            detail=f"A book with filename '{safe_filename}' already exists."
        )

    pdf_path = os.path.join(tempfile.gettempdir(), f"temp_{uuid.uuid4().hex[:8]}.pdf")
    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    job_id = uuid.uuid4().hex[:12]
    database.create_job(job_id, safe_filename)

    background_tasks.add_task(
        _run_ingest,
        job_id=job_id,
        pdf_path=pdf_path,
        original_filename=file.filename,
        safe_filename=safe_filename
    )

    return {
        "message": "Book upload received. Processing in background.",
        "job_id": job_id,
        "status": "pending"
    }

@app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse, dependencies=[Depends(verify_api_key)])
def get_job_status(job_id: str):
    job = database.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job

@app.delete("/v1/books/{book_id}", response_model=DeleteResponse, dependencies=[Depends(verify_api_key)])
def delete_book(book_id: str):
    deleted = database.delete_book(book_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Book '{book_id}' not found.")
    return {
        "message": f"Book '{book_id}' and all its data has been permanently deleted.",
        "book_id": book_id
    }

@app.post("/v1/query", response_model=QueryResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit("20/minute")
def query_book(request: Request, req: QueryRequest, db: Session = Depends(database.get_db)):
    if not database.user_exists(req.user_id):
        raise HTTPException(status_code=403, detail=f"Unknown user_id '{req.user_id}'. Register first.")

    book_row = db.execute(
        text("SELECT title FROM books WHERE id = :id"),
        {"id": req.book_id}
    ).mappings().fetchone()

    if not book_row:
        raise HTTPException(status_code=404, detail=f"Book '{req.book_id}' not found.")

    book_title = book_row["title"]
    history = database.get_chat_history(req.user_id, req.book_id)

    class MemoryWrapper:
        def get_context(self, limit=6): return history

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

    answer = generate_answer(
        query=req.query,
        context_chunks=chunks_text,
        memory=MemoryWrapper(),
        book_title=book_title
    )

    database.log_message(req.user_id, req.book_id, "user", req.query, req.chapter_limit)
    database.log_message(req.user_id, req.book_id, "bot", answer, req.chapter_limit)

    return {"answer": answer, "sources": sources}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)