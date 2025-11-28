# server/Dockerfile

FROM python:3.13-slim

# 1. Install system dependencies in a single, clean layer
# Note: This is an improved version of your existing RUN block.
# We ensure tesseract-ocr is installed.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 && \
    rm -rf /var/lib/apt/lists/*

# 2. Set the working directory
WORKDIR /app

# 3. Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of the application code
COPY . .

# 5. Run the application
# Render provides the port in the $PORT environment variable, but you used 10000 in your logs.
# This command binds to the port properly.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 10000"]