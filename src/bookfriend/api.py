import os
import shutil
import uuid
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Header
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
    book_id: str
    title: str

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
def ingest_book(file: UploadFile = File(...)):
    safe_filename = file.filename.replace(" ", "_")
    pdf_path = f"/tmp/temp_{uuid.uuid4().hex[:8]}.pdf"

    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"⚙️ Processing Book ({safe_filename})...")

    try:
        book_id = database.register_book(
            title=file.filename,
            filename=safe_filename,
            index_path="supabase-pgvector"
        )
        process_and_ingest_pdf(pdf_path, book_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

    return {"message": "Book processed and stored in Supabase pgvector", "book_id": book_id, "title": file.filename}


@app.post("/v1/query", response_model=QueryResponse, dependencies=[Depends(verify_api_key)])
def query_book(req: QueryRequest, db: Session = Depends(database.get_db)):
    # 1. Look up the real book title so the AI knows what it's talking about
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

    # 3. Semantic search (Spoiler Shield applied inside)
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

    # 4. Generate answer — now passes the real book title dynamically
    answer = generate_answer(
        query=req.query,
        context_chunks=chunks_text,
        memory=MemoryWrapper(),
        book_title=book_title        # ← the fix
    )

    # 5. Log to history
    database.log_message(req.user_id, req.book_id, "user", req.query, req.chapter_limit)
    database.log_message(req.user_id, req.book_id, "bot", answer, req.chapter_limit)

    return {"answer": answer, "sources": sources}