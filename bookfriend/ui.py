import streamlit as st
import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()

# --- Config ---
st.set_page_config(page_title="BookFriend", page_icon="📘", layout="wide")

API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("BOOKFRIEND_API_KEY", "bookfriend1234567apikey")
HEADERS = {"x-api-key": API_KEY}

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "selected_book_id" not in st.session_state:
    st.session_state.selected_book_id = None
if "books" not in st.session_state:
    st.session_state.books = []

# --- Helper Functions ---
def register_user():
    try:
        response = requests.post(f"{API_URL}/v1/register", headers=HEADERS)
        if response.status_code == 200:
            st.session_state.user_id = response.json()["user_id"]
            return True
    except Exception as e:
        st.error(f"Error connecting to backend: {e}")
    return False

def fetch_books():
    try:
        response = requests.get(f"{API_URL}/v1/books", headers=HEADERS)
        if response.status_code == 200:
            st.session_state.books = response.json()
    except Exception:
        pass

# --- UI Sidebar ---
with st.sidebar:
    st.title("📘 BookFriend")
    st.subheader("Your AI Reading Companion")

    if not st.session_state.user_id:
        if st.button("Register / New Session"):
            register_user()
            st.rerun()
    else:
        st.success(f"User: {st.session_state.user_id[:8]}...")

    st.divider()

    # Book Selection
    fetch_books()
    book_titles = {b["id"]: b["title"] for b in st.session_state.books}
    if book_titles:
        selected_id = st.selectbox(
            "Select a Book",
            options=list(book_titles.keys()),
            format_func=lambda x: book_titles[x],
            index=0 if not st.session_state.selected_book_id else list(book_titles.keys()).index(st.session_state.selected_book_id)
        )
        if selected_id != st.session_state.selected_book_id:
            st.session_state.selected_book_id = selected_id
            st.session_state.messages = [] # Clear chat when switching books
            st.rerun()
    else:
        st.info("No books uploaded yet.")

    st.divider()

    # Settings
    st.subheader("Spoiler Shield")
    chapter_limit = st.slider("Max Chapter to Search", 0, 100, 20)

    st.divider()

    # Upload
    st.subheader("Upload New Book")
    uploaded_file = st.file_uploader("PDF or EPUB", type=["pdf", "epub"])
    book_title = st.text_input("Book Title", placeholder="Enter book title...")

    if st.button("Ingest Book") and uploaded_file and book_title:
        with st.spinner("Uploading and starting ingestion..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            data = {"title": book_title}
            res = requests.post(f"{API_URL}/v1/upload", headers=HEADERS, files=files, data=data)
            if res.status_code == 200:
                st.success(f"Job started! ID: {res.json()['job_id']}")
                fetch_books()
            else:
                st.error("Upload failed.")

# --- Main Chat Area ---
if not st.session_state.selected_book_id:
    st.title("Welcome to BookFriend")
    st.info("Please select a book from the sidebar or upload a new one to start chatting.")
else:
    current_book_title = book_titles.get(st.session_state.selected_book_id, "Book")
    st.title(f"Chatting about: {current_book_title}")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask anything about the book..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching and thinking..."):
                payload = {
                    "user_id": st.session_state.user_id,
                    "book_id": st.session_state.selected_book_id,
                    "query": prompt,
                    "chapter_limit": chapter_limit
                }
                try:
                    response = requests.post(f"{API_URL}/v1/query", headers=HEADERS, json=payload)
                    if response.status_code == 200:
                        full_res = response.json()
                        answer = full_res["answer"]
                        st.markdown(answer)

                        if full_res.get("sources"):
                            with st.expander("View Sources"):
                                for src in full_res["sources"]:
                                    st.caption(f"- {src}")

                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    else:
                        error_msg = response.json().get("detail", "Unknown error")
                        st.error(f"API Error: {error_msg}")
                except Exception as e:
                    st.error(f"Error: {e}")

# --- Custom Styling ---
st.markdown("""
<style>
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stChatInputContainer {
        padding-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)
