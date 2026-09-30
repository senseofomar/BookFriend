import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL)

def drop():
    with engine.connect() as conn:
        print("Dropping book_chunks table...")
        conn.execute(text("DROP TABLE IF EXISTS book_chunks"))
        conn.commit()
        print("Done. It will be recreated with correct dimensions on next API start.")

if __name__ == "__main__":
    drop()
