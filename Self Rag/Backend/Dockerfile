# HuggingFace Spaces requires port 7860
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# jq-dev: needed by the Python 'jq' package (JSONLoader)
# gcc/g++: needed to compile some Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libjq-dev \
    libonig-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker cache optimization)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files (including Vector_store/)
COPY . .

# HuggingFace Spaces uses port 7860
EXPOSE 7860

# Start FastAPI with uvicorn
CMD ["uvicorn", "fastAPI:app", "--host", "0.0.0.0", "--port", "7860"]
