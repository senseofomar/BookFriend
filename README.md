---
title: BookFriend
emoji: 📘
colorFrom: blue
colorTo: indigo
sdk: streamlit
app_file: app.py
pinned: false
---

# 📘 BookFriend

**An API-first, spoiler-aware AI reading assistant.**

This project is optimized for deployment on **Streamlit Community Cloud**. It features a unified architecture where the interface and AI logic live in a single application.

---

## 🚀 How to Deploy (100% Free, No Credit Card)

1. **Push to GitHub**: Push this repository to your own GitHub account.
2. **Go to Streamlit**: Log in to [share.streamlit.io](https://share.streamlit.io) using your GitHub account.
3. **Deploy App**:
   * Repository: `your-username/bookfriend`
   * Branch: `main`
   * Main file path: `app.py`
4. **Set Secrets**: Before clicking deploy, click on **Advanced settings...** and paste your API keys into the **Secrets** box:
   ```toml
   DATABASE_URL = "your_supabase_postgresql_url"
   GEMINI_API_KEY = "your_gemini_key"
   GROQ_API_KEY = "your_groq_key"
   ```
5. **Deploy**: Click Deploy! You'll get a public link to share.

---

## 💻 How to Run Locally

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Environment Variables**: Make sure your `.env` file is present in the root directory.
3. **Start the App**:
   ```bash
   streamlit run app.py
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
| **Framework** | Streamlit |
| **Embeddings** | Gemini (`gemini-embedding-001`) |
| **LLM** | Groq (`llama-3.3-70b-versatile`) |
| **Database** | Supabase (PostgreSQL + pgvector) |

---

## 👋 Author

**[senseofomar]**
