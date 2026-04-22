# Use lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (for PyMuPDF, FAISS, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Create required directories
RUN mkdir -p data/papers data/indices evaluation

# Expose ports
EXPOSE 8000
EXPOSE 8501

# Start both services
RUN chmod +x start.sh
CMD ["bash", "start.sh"]