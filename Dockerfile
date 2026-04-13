FROM python:3.10-slim

# Install FFmpeg, curl, unzip, and runtime libraries required by ML dependencies
RUN apt-get update && apt-get install -y ffmpeg libgomp1 curl unzip ca-certificates && rm -rf /var/lib/apt/lists/*

# Install Deno for yt-dlp YouTube challenge solving
RUN curl -fsSL https://deno.land/install.sh | sh -s -- -y
ENV PATH="/root/.deno/bin:${PATH}"

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY server.py .
COPY static ./static

# Expose the port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
