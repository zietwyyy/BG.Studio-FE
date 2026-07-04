FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=7860 \
    HF_HOME=/tmp/huggingface

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create writable cache directory for Hugging Face
RUN mkdir -p /tmp/huggingface && chmod -R 777 /tmp/huggingface

# Copy requirements file
COPY requirements.txt .

# Install Python packages
# Install PyTorch with CUDA support for GPU instances
RUN pip install --no-cache-dir torch torchvision && \
    pip install --no-cache-dir -r requirements.txt


# Copy server code
COPY server.py .

# Expose Hugging Face default port
EXPOSE 7860

# Run FastAPI backend
CMD ["python", "server.py"]

