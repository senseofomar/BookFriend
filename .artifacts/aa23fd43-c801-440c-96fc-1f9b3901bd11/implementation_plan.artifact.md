# Implementation Plan: Enhancing BookFriend RAG App (Supabase & Render)

This plan addresses book format support, storage clarification, UI modernization, and file recovery for the BookFriend RAG application using **Supabase**, **Render**, and **Streamlit**.

## User Review Required

> [!IMPORTANT]
> **UI Choice**: Since Node.js/npm is not available in the current environment, I am proposing a **Streamlit** UI. It's built in Python, integrates perfectly with your existing code, and provides a clean "ChatGPT-style" chat interface that is easy to deploy on Render.
>
> **Database Connectivity**: Ensure your `DATABASE_URL` in `.env` is the transaction pooler URL from Supabase (Project Settings -> Database -> Connection string -> URI -> Pooler).

## Proposed Changes

### 1. Book Format Support (EPUB)
Currently, only PDF is supported. We will add EPUB support.

#### [MODIFY] [requirements.txt](file:///D:/PycharmProjects/bookfriend/requirements.txt)
* Add `ebooklib`, `beautifulsoup4`, and `streamlit`.

#### [MODIFY] [ingest.py](file:///D:/PycharmProjects/bookfriend/bookfriend/ingest.py)
* [DONE] Implemented `process_and_ingest_epub`.

#### [MODIFY] [api.py](file:///D:/PycharmProjects/bookfriend/bookfriend/api.py)
* [DONE] Updated with full FastAPI implementation and `/v1/upload` support.

### 2. Modern Chat UI (Streamlit)
We will create a professional chat interface in `bookfriend/ui.py`.
* **Features**:
    * Sidebar for book selection and uploading new books.
    * Chapter limit slider for "Spoiler Shield".
    * Summary toggle for quick recaps.
    * Persistent chat history using the backend API.

### 3. Database Schema Update
* [DONE] Updated `models.py` to include `User` and `IngestJob` tables.

### 4. Summarizer Recovery
* [DONE] Verified `summarizer.py` is present and functional.

## Verification Plan

### Automated Tests
*   Run ingestion tests for both PDF and EPUB files.
*   Verify semantic search returns relevant chunks from the database.

### Manual Verification
*   Run `streamlit run bookfriend/ui.py` locally.
*   Test the full flow: Upload EPUB -> Select Book -> Ask Question -> Get Answer.
