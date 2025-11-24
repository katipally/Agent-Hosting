# Production Dockerfile for Workforce Agent backend
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (build tools, curl for health checks if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (runtime-only; tests/dev deps are excluded)
COPY requirements-runtime.txt ./requirements-runtime.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-runtime.txt

# Copy backend source code
COPY . .

# Expose API port (matches Config.API_PORT default 8000)
EXPOSE 8000

# Default command: run FastAPI app via Uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
