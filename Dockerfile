# syntax=docker/dockerfile:1.7
# Reproducible image: base pinned by digest, dependencies from uv.lock, weights verified by sha256 at build time.
FROM python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e

# opencv-python (pulled by ultralytics) needs libGL/glib at import time.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.10.9 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_HTTP_TIMEOUT=300

WORKDIR /app
# Dependencies first: this layer is rebuilt only when the lockfile changes.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-install-project

# Build fails if the downloaded weights do not match the pinned sha256.
ADD --checksum=sha256:646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b \
    https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt /opt/weights/yolo26s.pt

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-editable

ARG GIT_COMMIT=unknown
ENV VIDINFER_GIT_COMMIT=$GIT_COMMIT \
    VIDINFER_WEIGHTS_DIR=/opt/weights \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib \
    PATH="/app/.venv/bin:$PATH"

RUN useradd --uid 1000 --create-home app
USER app
WORKDIR /work
ENTRYPOINT ["python", "-m", "vidinfer"]
CMD ["--help"]
