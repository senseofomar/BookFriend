import os
import google.generativeai as genai
from sqlalchemy import text
from dotenv import load_dotenv
from db import database

load_dotenv()

# Configure Gemini with environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

EMBEDDING_MODEL = "models/text-embedding-004"


def get_embedding(text_str: str) -> list:
    """Generates a 768-dim vector embedding using Google Gemini API."""
    response = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text_str,
        task_type="retrieval_document"
    )
    return response["embedding"]


def get_embeddings_batch(chunks: list) -> list:
    """Generates embeddings for a batch of text chunks using Google Gemini API."""
    response = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=chunks,
        task_type="retrieval_document"
    )
    return response["embedding"]


def upsert_book_to_supabase(book_id: str, chunks: list, chapters: list):
    """Embeds chunks via Gemini API and pushes them to Supabase pgvector."""
    print(f"🚀 Preparing {len(chunks)} chunks for Supabase upload via Gemini API...")

    # Fetch embeddings in batches
    embeddings = get_embeddings_batch(chunks)

    db = database.SessionLocal()
    try:
        query = text("""
            INSERT INTO book_chunks (book_id, chapter_num, chunk_text, embedding)
            VALUES (:book_id, :chapter_num, :chunk_text, CAST(:embedding AS vector))
        """)

        params = [
            {
                "book_id": book_id,
                "chapter_num": chapter,
                "chunk_text": chunk,
                "embedding": str(emb)  # String formatted list e.g. '[0.1, 0.2, ...]'
            }
            for chunk, chapter, emb in zip(chunks, chapters, embeddings)
        ]

        db.execute(query, params)
        db.commit()
        print(f"✅ Uploaded {len(chunks)} vectors to Supabase for book {book_id}")
    except Exception as e:
        db.rollback()
        print(f"❌ Error uploading to Supabase: {e}")
        raise
    finally:
        db.close()


def semantic_search(query: str, book_id: str, chapter_limit: int = None, top_k: int = 5):
    """Queries Supabase pgvector using cosine distance with Spoiler Shield."""
    query_vec = get_embedding(query)

    db = database.SessionLocal()
    try:
        if chapter_limit is not None:
            sql = text("""
                SELECT chunk_text, chapter_num,
                       1 - (embedding <=> CAST(:embedding AS vector)) AS similarity_score
                FROM book_chunks
                WHERE book_id = :book_id
                  AND chapter_num <= :chapter_limit
                ORDER BY embedding <=> CAST(:embedding AS vector)
                LIMIT :top_k
            """)
            params = {
                "embedding": str(query_vec),
                "book_id": book_id,
                "chapter_limit": chapter_limit,
                "top_k": top_k
            }
        else:
            sql = text("""
                SELECT chunk_text, chapter_num,
                       1 - (embedding <=> CAST(:embedding AS vector)) AS similarity_score
                FROM book_chunks
                WHERE book_id = :book_id
                ORDER BY embedding <=> CAST(:embedding AS vector)
                LIMIT :top_k
            """)
            params = {
                "embedding": str(query_vec),
                "book_id": book_id,
                "top_k": top_k
            }

        results = db.execute(sql, params).mappings().fetchall()
        return [(f"chapter_{row['chapter_num']}", row['chunk_text'], row['similarity_score']) for row in results]
    finally:
        db.close()