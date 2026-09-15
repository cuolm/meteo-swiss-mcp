FROM python:3.11-slim

# Copy uv binary from official image
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /bin/

WORKDIR /app

# Install system libraries required by rasterio/pyproj (GDAL) and meteodata-lab (ecCodes)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gdal-bin \
        libgdal-dev \
        libeccodes0 \
        libeccodes-dev && \
    rm -rf /var/lib/apt/lists/*

# ARG accepts dynamic versions from CI/CD (e.g., --build-arg BUILD_VERSION=1.2.0).
# Defaults to "0.0.0" for local builds.
ARG BUILD_VERSION=0.0.0

# Hatchling uses hatch-vcs, which delegates Git tag resolution to setuptools_scm under the hood.
# Because .git/ is excluded in .dockerignore (to reduce build context size and maximize cache hits),
# setuptools_scm fails to read repository tags. Setting SETUPTOOLS_SCM_PRETEND_VERSION overrides
# this check and sets the package version to $BUILD_VERSION during `uv sync`.
# In GitHub Actions, BUILD_VERSION is passed dynamically at build time.
ENV SETUPTOOLS_SCM_PRETEND_VERSION=$BUILD_VERSION

# Forces Python to print logs immediately so they appear in real-time without buffering
ENV PYTHONUNBUFFERED=1

# Cache layer: install third-party dependencies first
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --no-dev --frozen

# Copy source code and perform final package installation
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --frozen

# Activate the project virtual environment globally
ENV PATH="/app/.venv/bin:$PATH"

# Port used by the streamable-http transport
EXPOSE 8050

CMD ["meteo-swiss-mcp-server", "--transport=streamable-http", "--host=0.0.0.0", "--port=8050"]
