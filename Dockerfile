# 1. Base image
FROM python:3.10-slim

# 2. Env setup
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 4. Install Python deps first (layer cache)
WORKDIR /app
COPY src/bookfriend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Pre-download the embedding model into the image
#    Baked in at build time — zero cold-start delay in prod
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# 6. Copy source code
COPY src/bookfriend/ .

# 7. Expose port and run
#    $PORT is injected by Render at runtime (defaults to 8000 locally)
EXPOSE 8000
CMD uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}