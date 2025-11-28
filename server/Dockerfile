# Use the official Python 3.13 slim image (matches your current version)
FROM python:3.13-slim

# 1. Install system dependencies
# We need tesseract-ocr for text extraction
# We usually need ffmpeg or libsm6/libxext6 for OpenCV (even headless versions sometimes)
# server/Dockerfile
FROM python:3.13-slim

# 1. Install system dependencies
# Force install Tesseract-OCR, its dev libraries, and run-time libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    && rm -rf /var/lib/apt/lists/*

# ... (rest of your Dockerfile remains the same)

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