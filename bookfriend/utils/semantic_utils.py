import os
from google import genai
from sqlalchemy import text
from dotenv import load_dotenv

# Fixed import path
from bookfriend.db import database

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# Recommended model identifier for google-genai SDK
EMBEDDING_MODEL = "gemini-embedding-001"


def get_embeddings(texts: list) -> list:
    """Generates a list of embeddings for a batch of strings."""
    if not client:
        raise ValueError("GEMINI_API_KEY is not set.")

    # Using batch embedding capability
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
    )
    return [e.values for e in response.embeddings]


def get_embedding(text_str: str) -> list:
    """Generates a single embedding."""
    return get_embeddings([text_str])[0]


def upsert_book_to_supabase(book_id: str, chunks: list, chapters: list, batch_size: int = 50):
    """Embeds chunks in batches via Gemini API and pushes them to Supabase pgvector."""
    print(f"🚀 Preparing {len(chunks)} chunks for Supabase upload (Batch size: {batch_size})...")

    db = database.SessionLocal()
    try:
        query = text("""
            INSERT INTO book_chunks (book_id, chapter_num, chunk_text, embedding)
            VALUES (:book_id, :chapter_num, :chunk_text, CAST(:embedding AS vector))
        """)

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_chapters = chapters[i : i + batch_size]

            print(f"  → Processing batch {i // batch_size + 1}/{(len(chunks) - 1) // batch_size + 1}...")
            embeddings = get_embeddings(batch_chunks)

            params = []
            for chunk, chapter, emb in zip(batch_chunks, batch_chapters, embeddings):
                params.append({
                    "book_id": book_id,
                    "chapter_num": chapter,
                    "chunk_text": chunk,
                    "embedding": str(emb)
                })

            db.execute(query, params)
            db.commit()

        print(f"✅ Successfully uploaded {len(chunks)} vectors to Supabase.")
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