FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages from requirements.txt
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application files
COPY simple_rag.py .
COPY api.py .
COPY index.html .

# Create required directories with proper permissions
RUN mkdir -p legal_documents legal_vectordb && \
    chmod -R 777 /app/legal_documents /app/legal_vectordb

# Expose the API port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]