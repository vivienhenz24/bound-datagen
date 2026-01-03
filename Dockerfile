# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Install build dependencies for Rust
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain (version will be set by rust-toolchain.toml)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy codex-rs directory for building codex-exec
COPY codex-rs/ ./codex-rs/

# Build codex-exec binary
WORKDIR /app/codex-rs
RUN cargo build --release -p codex-exec

# Copy the built binary to /usr/local/bin
RUN cp target/release/codex-exec /usr/local/bin/codex-exec && \
    chmod +x /usr/local/bin/codex-exec

# Return to app directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY datagen/ ./datagen/

# Install dependencies using uv
RUN uv sync --no-dev

# Set a default command (can be overridden)
CMD ["python", "-c", "print('Docker container is running. Add your application code here.')"]

