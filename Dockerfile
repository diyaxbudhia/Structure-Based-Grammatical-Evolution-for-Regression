FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy repository into the image
COPY . /app

# Create output directories
RUN mkdir -p /app/outputs /app/figures

# Default command: run one short experiment (override via docker run args)
CMD ["python3", "ge_load_forecasting.py"]
