# 1. Use an official lightweight Python runtime
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Prevent Python from writing pyc files to disc and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 4. Install system dependencies (needed for some python packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5. Install Python dependencies
# We copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the source code into the container
COPY src/ .

# 7. Create a folder for database/uploads so they have permissions
RUN mkdir -p /app/bookfriend

# 8. Expose the port the app runs on
EXPOSE 8000

# 9. "cd" into the correct folder where api.py lives
WORKDIR /app/bookfriend

# 10. Run the application exactly like you did locally
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]