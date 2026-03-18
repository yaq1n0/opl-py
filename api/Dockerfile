# --- Builder stage ---
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Copy project files needed for pip install
COPY pyproject.toml README.md ./
COPY src/ src/

# Install opl-py with analytics dependencies (scikit-learn, polars, torch)
# Use CPU-only PyTorch to keep image smaller
ARG INSTALL_TORCH=true
RUN if [ "$INSTALL_TORCH" = "true" ]; then \
      pip install --no-cache-dir ".[analytics]" \
        --extra-index-url https://download.pytorch.org/whl/cpu; \
    else \
      pip install --no-cache-dir . scikit-learn polars; \
    fi

# Install API dependencies
COPY api/requirements.txt /build/api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

# --- Runtime stage ---
FROM python:3.12-slim

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy API source
COPY api/main.py /app/api/main.py

# Copy pre-built data (DB and pretrained models)
COPY data/opl.duckdb /app/data/opl.duckdb
COPY pretrained/ /app/pretrained/

# Environment
ENV OPL_DB_PATH=/app/data/opl.duckdb
ENV OPL_PRETRAINED_DIR=/app/pretrained

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
