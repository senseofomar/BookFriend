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

    # Generate embeddings
    embeddings = SEM_MODEL.encode(chunks, convert_to_numpy=True).tolist()

    db = database.SessionLocal()
    try:
        # Batch insert using raw SQL for maximum speed
        query = text("""
                     INSERT INTO book_chunks (book_id, chapter_num, chunk_text, embedding)
                     VALUES (:book_id, :chapter_num, :chunk_text, :embedding)
                     """)

        params = [
            {
                "book_id": book_id,
                "chapter_num": chapter,
                "chunk_text": chunk,
                "embedding": str(emb)  # pgvector accepts string representation of arrays
            }
            for chunk, chapter, emb in zip(chunks, chapters, embeddings)
        ]

        db.execute(query, params)
        db.commit()
        print(f"✅ Successfully uploaded {len(chunks)} vectors to Supabase for book {book_id}")
    except Exception as e:
        db.rollback()
        print(f"❌ Error uploading to Supabase: {e}")
        raise e
    finally:
        db.close()


def semantic_search(query: str, book_id: str, chapter_limit: int = None, top_k: int = 5):
    """Queries Supabase pgvector directly using cosine distance (<=>)."""
    query_vec = SEM_MODEL.encode([query], convert_to_numpy=True).tolist()[0]

    db = database.SessionLocal()
    try:
        # EXACTLY the query your friend designed, complete with the Spoiler Shield!
        if chapter_limit is not None:
            sql = text("""
                       SELECT chunk_text,
                              chapter_num,
                              1 - (embedding <=> :embedding) AS similarity_score
                       FROM book_chunks
                       WHERE book_id = :book_id
                         AND chapter_num <= :chapter_limit
                       ORDER BY embedding <=> :embedding
                LIMIT :top_k
                       """)
            params = {"embedding": str(query_vec), "book_id": book_id, "chapter_limit": chapter_limit, "top_k": top_k}
        else:
            sql = text("""
                       SELECT chunk_text,
                              chapter_num,
                              1 - (embedding <=> :embedding) AS similarity_score
                       FROM book_chunks
                       WHERE book_id = :book_id
                       ORDER BY embedding <=> :embedding
                LIMIT :top_k
                       """)
            params = {"embedding": str(query_vec), "book_id": book_id, "top_k": top_k}

        results = db.execute(sql, params).mappings().fetchall()

        # Format for your generate_answer function
        formatted_results = []
        for row in results:
            formatted_results.append((f"chapter_{row['chapter_num']}", row['chunk_text'], row['similarity_score']))

        return formatted_results
    finally:
        db.close()