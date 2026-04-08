FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the port used by HF Spaces
EXPOSE 7860

# Run the FastAPI server
CMD ["python", "server.py"]
