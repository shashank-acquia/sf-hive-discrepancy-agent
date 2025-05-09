# Use official lightweight Python image
FROM python:3.10-slim

# Install system dependencies for compiling Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    libssl-dev \
    || (apt-get --fix-missing update && apt-get install -y gcc libffi-dev libssl-dev) \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt
RUN pip3 install --upgrade certifi

# Copy the rest of the app code
COPY . .

# Expose app port
EXPOSE 5000

# Start the app
CMD ["python", "app.py"]
