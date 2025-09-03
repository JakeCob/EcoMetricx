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

# Create startup script inline and set up user
RUN echo '#!/bin/bash\n\
if [ -z "$PORT" ]; then\n\
    PORT=8000\n\
fi\n\
echo "Starting EcoMetricx API on port $PORT"\n\
uvicorn main:app --host 0.0.0.0 --port $PORT' > start.sh && \
    chmod +x start.sh && \
    useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app

USER app

# Expose port
EXPOSE 8000

# Start the application using the startup script
CMD ["./start.sh"]