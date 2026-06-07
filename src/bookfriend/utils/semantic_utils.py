from sentence_transformers import SentenceTransformer
from sqlalchemy import text
from dotenv import load_dotenv
import database

load_dotenv()

print("🧠 Loading embedding model...")
SEM_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def upsert_book_to_supabase(book_id: str, chunks: list, chapters: list):
    """Embeds chunks and pushes them directly to Supabase pgvector."""
    print(f"🚀 Preparing {len(chunks)} chunks for Supabase upload...")

    embeddings = SEM_MODEL.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    ).tolist()   # list of lists — each inner list is a Python float[]

    db = database.SessionLocal()
    try:
        query = text("""
            INSERT INTO book_chunks (book_id, chapter_num, chunk_text, embedding)
            VALUES (:book_id, :chapter_num, :chunk_text, :embedding)
        """)

        params = [
            {
                "book_id": book_id,
                "chapter_num": chapter,
                "chunk_text": chunk,
                # ✅ FIX: Pass as proper list, NOT str(emb)
                # pgvector driver (pgvector Python package) handles list→vector type
                "embedding": emb
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
    query_vec = SEM_MODEL.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    ).tolist()[0]  # ✅ FIX: list, not str

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
                "embedding": str(query_vec),  # CAST workaround for SQLAlchemy text()
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