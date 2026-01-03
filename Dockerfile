# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./

# Install dependencies using uv
RUN uv sync --no-dev

# Set a default command (can be overridden)
CMD ["python", "-c", "print('Docker container is running. Add your application code here.')"]

