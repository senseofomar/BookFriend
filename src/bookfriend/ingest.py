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


def process_and_ingest_pdf(pdf_path: str, book_id: str):
    """Reads PDF, chunks it by chapter, and sends directly to Pinecone."""
    print(f"📖 Reading {pdf_path} into memory...")

    reader = PdfReader(pdf_path)
    full_text = "".join([page.extract_text() or "" for page in reader.pages])

    # Try splitting by "Chapter X"
    pattern = r'(Chapter\s+\d+)'
    raw_chapters = re.split(pattern, full_text, flags=re.IGNORECASE)

    all_chunks = []
    all_chapters = []

    if len(raw_chapters) > 1:
        for i in range(1, len(raw_chapters), 2):
            chapter_title = raw_chapters[i].strip()
            chapter_content = raw_chapters[i + 1].strip()

            if len(chapter_content) < 500:
                continue

            # Extract the actual integer chapter number for Pinecone filtering
            try:
                chap_num = int(re.search(r'\d+', chapter_title).group())
            except:
                chap_num = 0

            # Chunk the chapter
            chunks = smart_chunking(chapter_content)
            all_chunks.extend(chunks)
            all_chapters.extend([chap_num] * len(chunks))  # Tag every chunk with its chapter
    else:
        print("⚠️ No 'Chapter X' headings found. Saving full text as Chapter 0.")
        chunks = smart_chunking(full_text)
        all_chunks.extend(chunks)
        all_chapters.extend([0] * len(chunks))

    if not all_chunks:
        raise ValueError("No text could be extracted or chunked from the PDF.")

    upsert_book_to_supabase(book_id, all_chunks, all_chapters)