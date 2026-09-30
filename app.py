import streamlit as st
import os
import uuid
import tempfile
import shutil
from dotenv import load_dotenv

# Load local .env if present
load_dotenv()

# --- Internal Imports ---
from bookfriend import db as database
from bookfriend.ingest import process_and_ingest_pdf, process_and_ingest_epub
from bookfriend.utils.semantic_utils import semantic_search
from bookfriend.utils.answer_generator import generate_answer, detect_intent
from bookfriend.services.summarizer import generate_global_summary
from bookfriend.utils.faq_utils import get_faq_answer

# --- Streamlit Config ---
st.set_page_config(page_title="BookFriend", page_icon="📘", layout="wide")

# --- Initialize DB ---
@st.cache_resource
def init_db_connection():
    database.init_db()
    return True

init_db_connection()

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "selected_book_id" not in st.session_state:
    st.session_state.selected_book_id = None

# --- Helper Logic ---
def get_books():
    db = next(database.get_db())
    try:
        from sqlalchemy import text
        rows = db.execute(text("SELECT id, title, filename FROM books")).mappings().fetchall()
        return [dict(r) for r in rows]
    finally:
        db.close()

def export_chat_history(messages, book_title):
    md_content = f"# Chat History: {book_title}\n\n"
    for msg in messages:
        role = "👤 You" if msg["role"] == "user" else "🤖 BookFriend"
        md_content += f"### {role}\n{msg['content']}\n\n---\n\n"
    return md_content

# --- UI Sidebar ---
with st.sidebar:
    st.title("📘 BookFriend")
    st.subheader("Your AI Reading Companion")

    if not st.session_state.user_id:
        if st.button("Start New Session"):
            st.session_state.user_id = database.create_user()
            st.rerun()
    else:
        st.success(f"Session Active: {st.session_state.user_id[:8]}...")

    st.divider()

    # Book Selection
    books = get_books()
    book_titles = {b["id"]: b["title"] for b in books}
    if book_titles:
        selected_id = st.selectbox(
            "Select a Book",
            options=list(book_titles.keys()),
            format_func=lambda x: book_titles[x],
            index=0 if not st.session_state.selected_book_id or st.session_state.selected_book_id not in book_titles else list(book_titles.keys()).index(st.session_state.selected_book_id)
        )
        if selected_id != st.session_state.selected_book_id:
            st.session_state.selected_book_id = selected_id
            st.session_state.messages = [] # Clear UI chat history (not DB history)
            st.rerun()
    else:
        st.info("No books uploaded yet.")

    st.divider()

    # Settings
    st.subheader("Spoiler Shield")
    chapter_limit = st.slider("Max Chapter to Search", 0, 100, 20)

    st.divider()

    # Chat Management
    st.subheader("Chat Management")
    if st.session_state.messages and st.session_state.selected_book_id:
        current_book_title = book_titles.get(st.session_state.selected_book_id, "Unknown Book")
        chat_markdown = export_chat_history(st.session_state.messages, current_book_title)
        st.download_button(
            label="⬇️ Download Chat History",
            data=chat_markdown,
            file_name=f"bookfriend_chat_{current_book_title.replace(' ', '_')}.md",
            mime="text/markdown"
        )
        if st.button("🗑️ Clear Chat History", type="secondary"):
            database.clear_chat_history(st.session_state.user_id, st.session_state.selected_book_id)
            st.session_state.messages = []
            st.rerun()


    st.divider()

    # Upload
    st.subheader("Upload New Book")
    uploaded_file = st.file_uploader("PDF or EPUB", type=["pdf", "epub"])
    book_title_input = st.text_input("Book Title", placeholder="Enter book title...")

    if st.button("Ingest Book") and uploaded_file and book_title_input:
        progress_text = "Ingesting... This may take a minute for large books."
        my_bar = st.progress(0, text=progress_text)

        def update_progress(current, total):
            percent = current / total
            my_bar.progress(percent, text=f"Processing chunk batch {current}/{total}...")

        try:
            # Save to temp file
            temp_dir = tempfile.gettempdir()
            file_path = os.path.join(temp_dir, f"{uuid.uuid4().hex}_{uploaded_file.name}")
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Register Book
            book_id = database.register_book(book_title_input, uploaded_file.name, "supabase")

            # Ingest
            if uploaded_file.name.lower().endswith(".pdf"):
                process_and_ingest_pdf(file_path, book_id, progress_callback=update_progress)
            else:
                process_and_ingest_epub(file_path, book_id, progress_callback=update_progress)

            os.remove(file_path)
            my_bar.empty()
            st.success(f"✅ '{book_title_input}' ingested successfully!")
            st.session_state.selected_book_id = book_id
            st.rerun()
        except Exception as e:
            my_bar.empty()
            st.error(f"❌ Ingestion failed: {e}")

# --- Main Chat Area ---
if not st.session_state.selected_book_id:
    st.title("Welcome to BookFriend")
    st.info("Select a book in the sidebar or upload one to start.")
else:
    current_book_title = book_titles.get(st.session_state.selected_book_id, "Book")
    st.title(f"Reading: {current_book_title}")

    # Load History from DB if UI history is empty
    if not st.session_state.messages and st.session_state.user_id:
        db_history = database.get_chat_history(st.session_state.user_id, st.session_state.selected_book_id)
        st.session_state.messages = db_history

    # Display Chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Ask about the book..."):
        if not st.session_state.user_id:
            st.error("Please start a session in the sidebar first.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    # 1. FAQ
                    faq_answer = get_faq_answer(prompt)
                    if faq_answer:
                        answer_output = f"[FAQ] {faq_answer}"
                    else:
                        # 2. Intent Detection
                        intent = detect_intent(prompt)
                        if intent == "SUMMARY":
                            answer_output = generate_global_summary(st.session_state.selected_book_id, current_book_title, chapter_limit)
                        else:
                            # 3. RAG Flow
                            class MemoryWrapper:
                                def get_context(self, limit=6): return st.session_state.messages

                            results = semantic_search(prompt, st.session_state.selected_book_id, chapter_limit)
                            if not results:
                                answer_output = "I couldn't find relevant info up to this chapter. Try increasing your chapter limit!"
                            else:
                                chunks = [c[1] for c in results]
                                answer_output = generate_answer(prompt, chunks, MemoryWrapper(), current_book_title)

                    if isinstance(answer_output, str):
                        st.markdown(answer_output)
                        final_answer = answer_output
                    else:
                        final_answer = st.write_stream(answer_output)

                    st.session_state.messages.append({"role": "assistant", "content": final_answer})

                    # Log to DB
                    database.log_message(st.session_state.user_id, st.session_state.selected_book_id, "user", prompt, chapter_limit)
                    database.log_message(st.session_state.user_id, st.session_state.selected_book_id, "assistant", final_answer, chapter_limit)

# --- Custom Styling ---
st.markdown("""
<style>
    .stChatMessage { border-radius: 15px; padding: 10px; margin-bottom: 10px; }
    .stChatInputContainer { padding-bottom: 20px; }
</style>
""", unsafe_allow_html=True)
