# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Install build dependencies for Rust and git
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain (version will be set by rust-toolchain.toml)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy only Cargo files first for better layer caching
# This allows Docker to cache the dependency download layer separately
COPY codex-rs/Cargo.toml codex-rs/Cargo.lock codex-rs/rust-toolchain.toml ./codex-rs/

# Now copy the rest of the codex-rs source code
COPY codex-rs/ ./codex-rs/

# Build codex-exec binary with cache mounts and copy it in the same step
WORKDIR /app/codex-rs
RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cargo/git \
    --mount=type=cache,target=/app/codex-rs/target \
    cargo build --release -p codex-exec && \
    cp target/release/codex-exec /usr/local/bin/codex-exec && \
    chmod +x /usr/local/bin/codex-exec

# Return to app directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY datagen/ ./datagen/

# Ensure workspace directory exists
RUN mkdir -p /app/workspace

# Install dependencies using uv
RUN uv sync --no-dev

# Set a default command (can be overridden)
CMD ["python", "-c", "print('Docker container is running. Add your application code here.')"]

