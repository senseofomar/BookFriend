import os
from groq import Groq
from sqlalchemy import text
from dotenv import load_dotenv
from bookfriend.db import database

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_book_chunks_for_summary(book_id: str, chapter_limit: int = None):
    """Retrieves all chunks up to a chapter limit to build a global summary."""
    db = database.SessionLocal()
    try:
        sql = """
            SELECT chunk_text, chapter_num
            FROM book_chunks
            WHERE book_id = :book_id
        """
        params = {"book_id": book_id}

        if chapter_limit is not None:
            sql += " AND chapter_num <= :chapter_limit"
            params["chapter_limit"] = chapter_limit

        sql += " ORDER BY chapter_num ASC, id ASC"

        results = db.execute(text(sql), params).mappings().fetchall()
        return results
    finally:
        db.close()

def summarize_batch(chunks: list, book_title: str) -> str:
    """Summarizes a single batch of chunks."""
    context = "\n\n".join(chunks)
    prompt = (
        f"Summarize the following excerpts from the book '{book_title}'. "
        "Focus on key plot points, character developments, and themes. "
        "Keep the summary concise but informative.\n\n"
        f"Excerpts:\n{context}"
    )

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error in batch summary: {e}")
        return ""

def generate_global_summary(book_id: str, book_title: str, chapter_limit: int = None) -> str:
    """Implements a Map-Reduce flow to summarize the whole book (up to chapter_limit)."""
    print(f"🔍 Generating global summary for '{book_title}'...")

    rows = get_book_chunks_for_summary(book_id, chapter_limit)
    if not rows:
        return "No content found to summarize."

    # Map Step: Group chunks into batches (e.g., 10 chunks per batch)
    all_text = [row["chunk_text"] for row in rows]
    batch_size = 15
    partial_summaries = []

    for i in range(0, len(all_text), batch_size):
        batch = all_text[i : i + batch_size]
        print(f"  → Mapping batch {len(partial_summaries) + 1}...")
        summary = summarize_batch(batch, book_title)
        if summary:
            partial_summaries.append(summary)

    # Reduce Step: Combine partial summaries
    if not partial_summaries:
        return "Failed to generate partial summaries."

    print(f"  → Reducing {len(partial_summaries)} partial summaries...")
    combined_context = "\n\n---\n\n".join(partial_summaries)

    final_prompt = (
        f"The following are partial summaries of different sections of the book '{book_title}'. "
        "Synthesize them into a single, coherent, and comprehensive overview of the book's narrative so far. "
        "Organize it logically (e.g., Introduction, Key Events, Current Status).\n\n"
        f"Partial Summaries:\n{combined_context}"
    )

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": final_prompt}],
            temperature=0.5,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error in final reduction: {e}")
        return "Failed to synthesize the final summary."
