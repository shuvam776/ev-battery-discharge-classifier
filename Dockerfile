FROM python:3.11-slim

LABEL maintainer="PunarShakti Energy <tech@punarshakti.energy>"
LABEL description="PunarShakti AI — Multimodal EV Battery Grading System"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ backend/
COPY models/ models/

# Environment configuration
ENV PYTHONUTF8=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    FRONTEND_URL="*" \
    MODEL_VERSION=1.0.0 \
    LOG_LEVEL=INFO

COPY .env.example .env

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
