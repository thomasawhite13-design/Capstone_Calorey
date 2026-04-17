# Use slim Python to keep image small
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install build dependencies (required by some Python packages)
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
      build-essential \
      gcc \
      libc-dev \
      git \
      curl && \
    rm -rf /var/lib/apt/lists/*

# Copy Python dependency manifest
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Environment
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Expose Cloud Run port
EXPOSE 8080

# Run via Gunicorn for production
CMD exec gunicorn --bind :8080 --workers 1 --threads 8 --timeout 0 app:app
