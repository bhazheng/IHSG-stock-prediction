FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Ensure app and code directories are importable by Python
ENV PYTHONPATH=/app:/app/code

WORKDIR /app

# Install system dependencies required by some packages (e.g., prophet, xgboost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    curl \
    git \
    ca-certificates \
    libatlas-base-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt /app/requirements.txt

# Upgrade pip and install Python dependencies
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the application code
COPY . /app

# Expose Flask port
EXPOSE 5000

# Default command: run the Flask app. Uses the included `app/main.py`.
# If you prefer Gunicorn, add it to requirements.txt and replace this CMD accordingly.
CMD ["python", "-u", "app/main.py"]
