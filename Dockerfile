# 1. Base image
FROM python:3.10-slim

# 2. Env setup
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# 3. System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 4. Working directory
WORKDIR /app

# 5. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy source code
COPY . .

# 7. Expose port and run (Render uses $PORT)
CMD ["uvicorn", "bookfriend.api:app", "--host", "0.0.0.0", "--port", "8000"]
