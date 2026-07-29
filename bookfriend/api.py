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
from bookfriend.ingest import process_and_ingest_pdf

# ... (rest of the imports and setup)

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
        top_k=5  # Increased top_k for better context
    )

    chunks_text = [chunk for _, chunk, _ in raw_results]
    sources = [source for source, _, _ in raw_results]

    if not chunks_text:
        return {"answer": "I couldn't find anything about that in the book up to this chapter. Try increasing your chapter limit if you've read further!", "sources": []}

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