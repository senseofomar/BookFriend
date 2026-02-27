import os
import shutil
import uuid
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from sqlalchemy import text

# Load secrets
load_dotenv()

# Import core logic
from utils.semantic_utils import semantic_search
from utils.answer_generator import generate_answer
import database
import models
from ingest import process_and_ingest_pdf

# === Setup App ===
app = FastAPI(
    title="BookFriend API",
    version="3.0 (Cloud Serverless Edition)"
)

@app.on_event("startup")
def startup_event():
    # Initialize the tables in Supabase.
    # Because 'models' is imported, SQLAlchemy knows what to build.
    database.init_db()
    print("✅ Application startup complete. Connected to Supabase.")
    # Notice we deleted all the FAISS loading logic here! 🧠

# === API Models ===

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

# === Endpoints ===

@app.get("/health")
def health_check():
    return {"status": "online", "db": "connected"}

@app.get("/v1/books", response_model=List[BookListResponse])
def list_books():
    """Returns a list of all ingested books so the frontend knows what IDs to use."""
    db_generator = database.get_db()
    conn = next(db_generator)
    try:
        # Wrapped in text() and using .mappings() to fix SQLAlchemy 2.0 errors
        rows = conn.execute(text("SELECT id, title, filename FROM books")).mappings().fetchall()
    finally:
        conn.close()

    return [
        {"id": r["id"], "title": r["title"], "filename": r["filename"]}
        for r in rows
    ]

@app.post("/v1/ingest", response_model=IngestResponse)
def ingest_book(file: UploadFile = File(...)):
    safe_filename = file.filename.replace(" ", "_")

    # Save PDF temporarily to disk just so PyPDF can read it
    pdf_path = f"temp_{uuid.uuid4().hex[:8]}.pdf"
    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"⚙️ Processing Book ({safe_filename})...")

    try:
        # 1. Register in Supabase first to get our unique book_id
        # We pass "pinecone" as the index path because local files are gone!
        book_id = database.register_book(
            title=file.filename,
            filename=safe_filename,
            index_path="pinecone"
        )

        # 2. Extract, Chunk, and Push to Pinecone directly in Python (No more subprocess!)
        process_and_ingest_pdf(pdf_path, book_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        # Clean up the temporary PDF. The server leaves no trace behind!
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

    return {"message": "Book Processed and Pushed to Pinecone", "book_id": book_id, "title": file.filename}

@app.post("/v1/query", response_model=QueryResponse)
def query_book(req: QueryRequest):
    # 1. History & Search
    history = database.get_chat_history(req.user_id, req.book_id)

    class MemoryWrapper:
        def get_context(self, limit=6): return history

    memory_mock = MemoryWrapper()

    # 2. Semantic Search (Pinecone natively applies the Spoiler Shield via metadata!)
    raw_results = semantic_search(
        query=req.query,
        book_id=req.book_id,
        chapter_limit=req.chapter_limit,
        top_k=3
    )

    # 3. Extract chunks
    chunks_text = [chunk for _, chunk, _ in raw_results]
    sources = [source for source, _, _ in raw_results]

    if not chunks_text:
        return {"answer": "I couldn't find anything about that in the book up to this chapter.", "sources": []}

    # 4. Answer & Log
    answer = generate_answer(req.query, chunks_text, memory=memory_mock)

    database.log_message(req.user_id, req.book_id, "user", req.query, req.chapter_limit)
    database.log_message(req.user_id, req.book_id, "bot", answer, req.chapter_limit)

    return {"answer": answer, "sources": sources}