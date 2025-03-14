# Use a lightweight Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code to container
COPY app.py .

# Expose port 5000 and set the entrypoint
EXPOSE 5000
CMD ["python", "app.py"]
