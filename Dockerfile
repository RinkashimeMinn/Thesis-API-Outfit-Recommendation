# Use an official lightweight Python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /app

# Copy project files and the Google Cloud key
COPY . .  

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for Cloud Run
ENV PORT 8080

# Set environment variable for GCS credentials
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/thesisapi-458811-982bb3fab395.json

# Run with gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "--timeout", "600", "main:app"]
