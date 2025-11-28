FROM python:3.13-slim

# 1. Install system dependencies in a single, clean layer
# FIX: Replaced 'libgl1-mesa-glx' with 'libgl1' to resolve the 'no installation candidate' error.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libgl1 \
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
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 10000"]