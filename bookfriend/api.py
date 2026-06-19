import os
import shutil
import uuid
import tempfile
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Header, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from sqlalchemy import text
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

load_dotenv()

from bookfriend.utils.semantic_utils import semantic_search
from bookfriend.utils.answer_generator import generate_answer, detect_intent
from bookfriend.services.summarizer import generate_global_summary
from bookfriend.utils.faq_utils import get_faq_answer
from bookfriend import db as database
from bookfriend.ingest import process_and_ingest_pdf, process_and_ingest_epub

# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database tables
    database.init_db()
    yield

# --- Rate Limiter ---
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="BookFriend API", version="2.0.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Security ---
API_KEY = os.getenv("BOOKFRIEND_API_KEY", "bookfriend1234567apikey")

def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return x_api_key

# --- Schemas ---
class QueryRequest(BaseModel):
    user_id: str
    book_id: str
    query: str
    chapter_limit: Optional[int] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

class RegisterResponse(BaseModel):
    user_id: str

class BookInfo(BaseModel):
    id: str
    title: str
    filename: str

class UploadResponse(BaseModel):
    job_id: str
    message: str

# --- Background Task ---
def bg_ingest(job_id: str, file_path: str, filename: str, book_title: str):
    try:
        database.update_job(job_id, "processing")

        # Save book to DB metadata and get its unique ID
        actual_book_id = database.register_book(book_title, filename, "supabase")

        if filename.lower().endswith(".pdf"):
            process_and_ingest_pdf(file_path, actual_book_id)
        elif filename.lower().endswith(".epub"):
            process_and_ingest_epub(file_path, actual_book_id)
        else:
            raise ValueError("Unsupported file format")

        database.update_job(job_id, "completed", book_id=actual_book_id)
    except Exception as e:
        print(f"❌ Ingestion Error: {e}")
        database.update_job(job_id, "failed", error=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

# --- Endpoints ---

@app.get("/")
def health_check():
    return {"status": "ok", "service": "bookfriend-api"}

@app.post("/v1/register", response_model=RegisterResponse, dependencies=[Depends(verify_api_key)])
def register_user():
    user_id = database.create_user()
    return {"user_id": user_id}

@app.get("/v1/books", response_model=List[BookInfo], dependencies=[Depends(verify_api_key)])
def list_books(db: Session = Depends(database.get_db)):
    rows = db.execute(text("SELECT id, title, filename FROM books")).mappings().fetchall()
    return [dict(r) for r in rows]

@app.post("/v1/upload", response_model=UploadResponse, dependencies=[Depends(verify_api_key)])
async def upload_book(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = "Untitled Book"
):
    if not (file.filename.endswith(".pdf") or file.filename.endswith(".epub")):
        raise HTTPException(status_code=400, detail="Only .pdf and .epub are supported.")

    # Save to temp file
    temp_dir = tempfile.gettempdir()
    job_id = uuid.uuid4().hex[:12]
    file_path = os.path.join(temp_dir, f"{job_id}_{file.filename}")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    database.create_job(job_id, file.filename)
    background_tasks.add_task(bg_ingest, job_id, file_path, file.filename, title)

    return {"job_id": job_id, "message": "Upload successful. Ingestion started in background."}

@app.get("/v1/jobs/{job_id}", dependencies=[Depends(verify_api_key)])
def check_job(job_id: str):
    job = database.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.post("/v1/query", response_model=QueryResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
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

    # 1. Check FAQ first (low cost)
    faq_answer = get_faq_answer(req.query)
    if faq_answer:
        return {"answer": f"[FAQ] {faq_answer}", "sources": ["internal_faq"]}

    # 2. Detect Intent (SUMMARY vs QUERY)
    intent = detect_intent(req.query)
    print(f"DEBUG: Detected intent '{intent}' for query: {req.query}")

    if intent == "SUMMARY":
        answer = generate_global_summary(req.book_id, book_title, req.chapter_limit)
        database.log_message(req.user_id, req.book_id, "user", req.query, req.chapter_limit)
        database.log_message(req.user_id, req.book_id, "bot", answer, req.chapter_limit)
        return {"answer": answer, "sources": ["all_chapters_summarized"]}

    # 3. Standard RAG Flow
    history = database.get_chat_history(req.user_id, req.book_id)

    class MemoryWrapper:
        def get_context(self, limit=6): return history

    raw_results = semantic_search(
        query=req.query,
        book_id=req.book_id,
        chapter_limit=req.chapter_limit,
        top_k=5
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
