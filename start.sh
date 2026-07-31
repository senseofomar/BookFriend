#!/bin/bash

# Start the FastAPI backend in the background on port 8000
echo "🚀 Starting FastAPI backend..."
uvicorn bookfriend.api:app --host 0.0.0.0 --port 8000 &

# Start the Streamlit frontend on port 7860 (HF standard port)
echo "🎨 Starting Streamlit frontend..."
streamlit run bookfriend/ui.py --server.port 7860 --server.address 0.0.0.0
