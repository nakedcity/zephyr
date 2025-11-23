FROM python:3.10-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files
COPY pyproject.toml .
COPY config.yaml .

# Install dependencies
RUN uv sync --frozen

# Copy source code
COPY . .

# Create cache directory
RUN mkdir -p cache

# Expose port
EXPOSE 8080

# Run server
CMD ["uv", "run", "python", "-m", "server.main"]
