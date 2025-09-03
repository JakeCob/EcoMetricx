FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .

# Install Python dependencies without build tools (faster)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY server.py .

# Create non-root user for security  
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app

USER app

# Expose port
EXPOSE 8000

# Start the application using the server script
CMD ["python", "server.py"]