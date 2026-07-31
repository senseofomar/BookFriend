---
title: BookFriend
emoji: 📘
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# 📘 BookFriend

**An API-first, spoiler-aware AI reading assistant.**

This project is hosted on Hugging Face Spaces using Docker. It runs both a **FastAPI** backend and a **Streamlit** frontend in a single container.

---

## 🚀 How to Run Locally

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Start the API**:
   ```bash
   uvicorn bookfriend.api:app --reload
   ```
3. **Start the UI** (in a new terminal):
   ```bash
   streamlit run bookfriend/ui.py
   ```

---

## ✨ Key Features

*   **EPUB & PDF Support**: Upload and chat with both formats.
*   **Spoiler Shield**: Set a chapter limit to prevent the AI from revealing future events.
*   **Global Summaries**: Generate comprehensive recaps using Map-Reduce.
*   **Supabase Integration**: Uses `pgvector` for efficient semantic search.

---

## 📦 Tech Stack

| Component | Technology |
| :--- | :--- |
| **Backend** | FastAPI |
| **Frontend** | Streamlit |
| **Embeddings** | Gemini (`gemini-embedding-001`) |
| **LLM** | Groq (`llama-3.3-70b-versatile`) |
| **Database** | Supabase (PostgreSQL + pgvector) |

---

## 👋 Author

**[senseofomar]**
