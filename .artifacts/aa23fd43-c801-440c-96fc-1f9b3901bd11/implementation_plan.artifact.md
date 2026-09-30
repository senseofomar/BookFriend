# Implementation Plan: BookFriend Feature Expansion

While I cannot generate backdated commits to manipulate GitHub contribution graphs, I can certainly help you build a robust set of **real, highly requested features** for BookFriend. We will implement these step-by-step with proper, meaningful Git commits for each feature, representing authentic and professional development work.

## User Review Required

> [!IMPORTANT]
> **Streaming Responses**: I propose upgrading the AI response generation to use **Streaming**. Instead of the user waiting 5-10 seconds for a large answer to appear all at once, the text will type out in real-time (like ChatGPT).
>
> **Are you okay with these feature proposals?** Once you approve, I will implement them and commit them organically.

## Proposed New Features

### 1. Real-Time Streaming Responses (UX Upgrade)
Currently, the app blocks the UI while waiting for the Groq API to finish generating the answer.
*   **[MODIFY] `bookfriend/utils/answer_generator.py`**: Update the Groq completion call to use `stream=True` and yield tokens.
*   **[MODIFY] `app.py`**: Use Streamlit's `st.write_stream()` to display the text dynamically as it arrives.

### 2. Export Chat History (Utility)
Allow users to download their reading notes and Q&A sessions.
*   **[MODIFY] `app.py`**: Add a "Download Chat" button in the sidebar that compiles the current `st.session_state.messages` into a formatted Markdown or Text file for the user to save.

### 3. Ingestion Progress Tracking (Visibility)
Large EPUBs and PDFs take time to chunk and embed. A spinner isn't enough feedback.
*   **[MODIFY] `bookfriend/ingest.py`**: Add callback support to report chunking and embedding progress.
*   **[MODIFY] `app.py`**: Use Streamlit's `st.progress()` bar during the file upload process to show exactly how many chapters/chunks have been processed.

### 4. Chat Management (Quality of Life)
*   **[MODIFY] `app.py`**: Add a "Clear Chat" button to wipe the current conversation history from the database and UI, allowing the user to start fresh without creating a whole new session.

## Verification Plan

### Automated/Manual Verification
*   Test streaming by asking a complex question and ensuring the UI updates in real-time without freezing.
*   Verify the Markdown export contains the correct formatting and roles.
*   Upload a large EPUB and watch the progress bar increment correctly.
