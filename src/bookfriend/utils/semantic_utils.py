import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

print("🧠 Loading embedding model...")
# Load model globally so it only initializes once
SEM_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("CRITICAL: PINECONE_API_KEY is not set in .env")

# Connect to Pinecone
print("☁️ Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "bookfriend-index"
pinecone_index = pc.Index(INDEX_NAME)


def upsert_book_to_pinecone(book_id: str, chunks: list, chapters: list):
    """Embeds chunks and pushes them to the Pinecone cloud."""
    vectors = []
    print(f"🚀 Preparing {len(chunks)} chunks for Pinecone upload...")

    # Generate embeddings for all chunks at once (much faster)
    embeddings = SEM_MODEL.encode(chunks, convert_to_numpy=True).tolist()

    for i, (chunk, chapter, emb) in enumerate(zip(chunks, chapters, embeddings)):
        chunk_id = f"{book_id}_{i}"

        # Metadata is how we filter later!
        metadata = {
            "book_id": book_id,
            "chapter": chapter,
            "text": chunk
        }

        vectors.append((chunk_id, emb, metadata))

    # Batch upload (100 at a time)
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i: i + batch_size]
        pinecone_index.upsert(vectors=batch)

    print(f"✅ Successfully uploaded {len(chunks)} vectors to Pinecone for book {book_id}")


def semantic_search(query: str, book_id: str, chapter_limit: int = None, top_k: int = 5):
    """Queries Pinecone directly using metadata filters."""
    query_vec = SEM_MODEL.encode([query], convert_to_numpy=True).tolist()[0]

    # The Spoiler Shield Filter 🛡️
    filter_dict = {"book_id": {"$eq": book_id}}
    if chapter_limit is not None:
        filter_dict["chapter"] = {"$lte": chapter_limit}

    results = pinecone_index.query(
        vector=query_vec,
        top_k=top_k,
        include_metadata=True,
        filter=filter_dict
    )

    # Format the results to match what our answer generator expects
    formatted_results = []
    for match in results.get('matches', []):
        meta = match['metadata']
        # We mock the filename format just to keep the answer generator happy for now
        formatted_results.append((f"chapter_{meta['chapter']}", meta['text'], match['score']))

    return formatted_results