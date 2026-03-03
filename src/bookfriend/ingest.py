import re
from pypdf import PdfReader
from utils.semantic_utils import upsert_book_to_supabase


def smart_chunking(text, chunk_size=800, overlap_sentences=2):
    """Sentence-safe chunking with bounded size and semantic overlap."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = []

    def current_len():
        return sum(len(s) for s in current)

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence: continue

        if current_len() + len(sentence) > chunk_size:
            chunks.append(" ".join(current))
            overlap = current[-overlap_sentences:] if overlap_sentences > 0 else []
            current = overlap[:]
            while current_len() + len(sentence) > chunk_size and len(current) > 0:
                current.pop(0)
            current.append(sentence)
        else:
            current.append(sentence)

    if current:
        chunks.append(" ".join(current))
    return chunks


