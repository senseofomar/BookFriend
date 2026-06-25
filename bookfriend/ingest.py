import re
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from pypdf import PdfReader
from bookfriend.utils.semantic_utils import upsert_book_to_supabase
from bookfriend import db as database

def smart_chunking(text, chunk_size=800, overlap_sentences=2):
    """Sentence-safe chunking with bounded size and semantic overlap."""
    # Basic cleanup
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = []

    def current_len():
        return sum(len(s) for s in current)

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence: continue

        if current_len() + len(sentence) > chunk_size:
            if current:
                chunks.append(" ".join(current))
            overlap = current[-overlap_sentences:] if overlap_sentences > 0 else []
            current = overlap[:]
            # Ensure overlap doesn't exceed chunk_size
            while current and current_len() + len(sentence) > chunk_size:
                current.pop(0)
            current.append(sentence)
        else:
            current.append(sentence)

    if current:
        chunks.append(" ".join(current))
    return chunks

def process_and_ingest_pdf(pdf_path: str, book_id: str):
    """Reads PDF, chunks it by chapter, and upserts to Supabase pgvector."""
    print(f"📖 Reading PDF {pdf_path} into memory...")

    reader = PdfReader(pdf_path)
    full_text = "".join([page.extract_text() or "" for page in reader.pages])

    pattern = r'(Chapter\s+\d+)'
    raw_chapters = re.split(pattern, full_text, flags=re.IGNORECASE)

    all_chunks = []
    all_chapters = []

    if len(raw_chapters) > 1:
        # If split by "Chapter X", the first element is usually intro text
        intro = raw_chapters[0].strip()
        if intro:
            chunks = smart_chunking(intro)
            all_chunks.extend(chunks)
            all_chapters.extend([0] * len(chunks))

        for i in range(1, len(raw_chapters), 2):
            chapter_title = raw_chapters[i].strip()
            chapter_content = raw_chapters[i + 1].strip()

            try:
                chap_num = int(re.search(r'\d+', chapter_title).group())
            except Exception:
                chap_num = 0

            chunks = smart_chunking(chapter_content)
            all_chunks.extend(chunks)
            all_chapters.extend([chap_num] * len(chunks))
    else:
        print("⚠️ No 'Chapter X' headings found. Saving full text as Chapter 0.")
        chunks = smart_chunking(full_text)
        all_chunks.extend(chunks)
        all_chapters.extend([0] * len(chunks))

    if not all_chunks:
        raise ValueError("No text could be extracted or chunked from the PDF.")

    upsert_book_to_supabase(book_id, all_chunks, all_chapters)

def process_and_ingest_epub(epub_path: str, book_id: str):
    """Reads EPUB, extracts text by document item, and upserts to Supabase pgvector."""
    print(f"📖 Reading EPUB {epub_path} into memory...")

    book = epub.read_epub(epub_path)
    all_chunks = []
    all_chapters = []

    chapter_count = 0
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            text = soup.get_text()

            # Skip very short items (likely nav or empty)
            if len(text.strip()) < 100:
                continue

            chapter_count += 1
            chunks = smart_chunking(text)
            all_chunks.extend(chunks)
            all_chapters.extend([chapter_count] * len(chunks))

    if not all_chunks:
        raise ValueError("No text could be extracted or chunked from the EPUB.")

    print(f"✅ Extracted {len(all_chunks)} chunks from {chapter_count} items.")
    upsert_book_to_supabase(book_id, all_chunks, all_chapters)
