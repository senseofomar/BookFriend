import os
import shutil
import sys
import subprocess
from sqlalchemy import text
import database
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load secrets
load_dotenv()

# Import core logic
from utils.semantic_utils import load_semantic_index_from_path, semantic_search
from utils.answer_generator import generate_answer
import database  # <--- NEW: Import our DB module

# === Setup App ===
app = FastAPI(
    title="BookFriend API",
    version="2.2 (Multi-book)"
)

# === Global State (Multi-Tenant Cache) ===
class AppState:
    # Dictionary to store multiple books: { "book_id": (index, mapping) }
    indices: Dict[str, Any] = {}

state = AppState()

# === Helper: Load a specific book ===
def load_book_index(book_id: str, index_path: str):
    if book_id in state.indices:
        return

    if not os.path.exists(index_path):
        print(f"⚠️ Index missing for {book_id}")
        return

    print(f"📖 Loading Index for {book_id}...")
    # You need to update semantic_utils.py to accept a path!
    # (See Action 4 below. For now, we assume a helper exists)
    try:
        idx, mapping = load_semantic_index_from_path(index_path)
        state.indices[book_id] = (idx, mapping)
        print(f"✅ Loaded {book_id}")
    except Exception as e:
        print(f"❌ Failed to load {book_id}: {e}")


@app.on_event("startup")
def startup_event():
    # 1. Initialize the tables (which you already fixed)
    database.init_db()

    # 2. Manually grab the session out of the generator
    db_generator = database.get_db()
    conn = next(db_generator)

    try:
        # 3. SQLAlchemy requires raw SQL strings to be wrapped in text()
        books = conn.execute(text("SELECT id, index_path FROM books")).fetchall()

        # ... the rest of your startup logic for loading FAISS indexes ...

    finally:
        # Always close the connection when doing manual generation
        conn.close()

    for b in books:
        load_book_index(b["id"], b["index_path"])

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
    conn = database.get_db()
    rows = conn.execute("SELECT id, title, filename FROM books").fetchall()
    conn.close()

    return [
        {"id": r["id"], "title": r["title"], "filename": r["filename"]}
        for r in rows
    ]

@app.post("/v1/ingest", response_model=IngestResponse)
def ingest_book(file: UploadFile = File(...)):
    # 1. Generate IDs
    import uuid
    process_id = str(uuid.uuid4())[:8]

    # 2. SANITIZATION RESTORED: Clean the original name for the DB
    # We strip spaces and weird chars so the DB entry is clean
    safe_filename = file.filename.replace(" ", "_")

    # 3. Define Paths (Use process_id for disk storage to prevent overwrites)
    pdf_path = f"upload_{process_id}.pdf"
    chapters_dir = f"chapters_{process_id}"
    index_path = f"index_{process_id}.faiss"

    # Save to disk using the SAFE ID, not the filename
    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"⚙️ Processing Book {process_id} ({safe_filename})...")

    try:
        env = os.environ.copy()
        # Pass the UUID paths to the scripts
        subprocess.run([sys.executable, "ingest.py", pdf_path, chapters_dir], check=True, env=env)
        subprocess.run([sys.executable, "build_index.py", chapters_dir, index_path], check=True, env=env)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
    finally:
        if os.path.exists(pdf_path): os.remove(pdf_path)

    # 4. Register in DB
    # Use the 'safe_filename' here so the user sees a nice name
    book_id = database.register_book(
        title=file.filename,  # Original title (e.g., "Lord of Mysteries.pdf")
        filename=safe_filename,  # Cleaned filename (e.g., "Lord_of_Mysteries.pdf")
        index_path=index_path  # Points to the UUID index file
    )

    load_book_index(book_id, index_path)

    return {"message": "Book Processed", "book_id": book_id, "title": file.filename}


@app.post("/v1/query", response_model=QueryResponse)
def query_book(req: QueryRequest):
    # 1. Load Index if missing
    if req.book_id not in state.indices:
        conn = database.get_db()
        row = conn.execute("SELECT index_path FROM books WHERE id = ?", (req.book_id,)).fetchone()
        conn.close()
        if row:
            load_book_index(req.book_id, row["index_path"])

        if req.book_id not in state.indices:
            raise HTTPException(status_code=404, detail="Book not found or not indexed.")

    idx, mapping = state.indices[req.book_id]

    # 2. History & Search
    history = database.get_chat_history(req.user_id, req.book_id)

    class MemoryWrapper:
        def get_context(self, limit=6): return history

    memory_mock = MemoryWrapper()

    raw_results = semantic_search(req.query, idx, mapping)

    # 3. Filter Spoilers
    safe_results = []
    limit = req.chapter_limit
    for fname, chunk, _ in raw_results:
        try:
            chap_num = int(''.join(filter(str.isdigit, fname)))
            if chap_num <= limit: safe_results.append((fname, chunk))
        except:
            safe_results.append((fname, chunk))

    final_context = safe_results[:3]
    chunks_text = [c for _, c in final_context]

    # 4. Answer & Log
    answer = generate_answer(req.query, chunks_text, memory=memory_mock)

    database.log_message(req.user_id, req.book_id, "user", req.query, req.chapter_limit)
    database.log_message(req.user_id, req.book_id, "bot", answer, req.chapter_limit)

    return {"answer": answer, "sources": [f for f, _ in final_context]}