# Multi-stage build for optimized CI/CD with uv
FROM python:3.11-slim as builder

# Install uv - much faster than pip
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy dependency files first for better Docker layer caching
COPY requirements_complete.txt ./

# Install dependencies with uv (10-100x faster than pip)
RUN uv pip install --system -r requirements_complete.txt

# Production stage
FROM python:3.11-slim as production

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Create logs directory
RUN mkdir -p logs

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run Chainlit application
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8080"]
