# Implementation Plan: True Free Deployment (Streamlit Community Cloud)

Since Hugging Face and Render have changed their free tier policies, we will use **Streamlit Community Cloud**. It is 100% free, requires **no credit card**, and only needs a GitHub account to host your app.

## User Review Required

> [!IMPORTANT]
> **Unified App Architecture**: Streamlit Community Cloud hosts a single Python script. To make this work, I will merge your "brain" (API logic) directly into the "face" (UI script). You will no longer need to run two separate commands.
>
> **Secrets Management**: Instead of an `.env` file, you will paste your API keys into the "Secrets" settings on the Streamlit dashboard.

## Proposed Changes

### 1. Unified Application Logic
We will create a single, powerful `app.py` that contains both the logic and the interface.
*   **[NEW] [app.py](file:///D:/PycharmProjects/bookfriend/app.py)**: This will combine:
    *   Database initialization and models.
    *   Book ingestion (PDF/EPUB).
    *   Semantic search and RAG answering.
    *   The Streamlit Chat UI.

### 2. Configuration for Streamlit
*   **[MODIFY] [requirements.txt](file:///D:/PycharmProjects/bookfriend/requirements.txt)**: Ensure all necessary libraries for both ingestion and UI are listed.
*   **[DELETE] [Dockerfile](file:///D:/PycharmProjects/bookfriend/Dockerfile)**, **[render.yaml](file:///D:/PycharmProjects/bookfriend/render.yaml)**, **[start.sh](file:///D:/PycharmProjects/bookfriend/start.sh)**: These are no longer needed for Streamlit Community Cloud.

## Deployment Steps

1.  **GitHub**: Push your code to a GitHub repository (Public or Private).
2.  **Streamlit Share**: Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3.  **Deploy**:
    *   Select your repo and the `app.py` file.
    *   Click **"Advanced Settings"**.
    *   **Secrets**: Paste the contents of your `.env` file here in TOML format (I will provide the exact text).
4.  **Share**: You will get a link like `https://bookfriend.streamlit.app` to send to your friends.

## Verification Plan

### Manual Verification
*   Run `streamlit run app.py` locally to ensure the merged logic works before pushing.
*   Verify book upload and chat functionality in the deployed environment.
