import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL)

def check():
    with engine.connect() as conn:
        print("--- Books ---")
        books = conn.execute(text("SELECT id, title FROM books")).fetchall()
        for b in books:
            chunk_count = conn.execute(text("SELECT count(*) FROM book_chunks WHERE book_id = :id"), {"id": b[0]}).scalar()
            print(f"ID: {b[0]}, Title: {b[1]}, Chunks: {chunk_count}")

        print("\n--- Jobs ---")
        jobs = conn.execute(text("SELECT id, status, error FROM ingest_jobs ORDER BY created_at DESC LIMIT 5")).fetchall()
        for j in jobs:
            print(f"ID: {j[0]}, Status: {j[1]}, Error: {j[2]}")

if __name__ == "__main__":
    check()
