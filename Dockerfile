# Use Python 3.11 official image as base
FROM python:3.11

# Set working directory inside container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    build-essential \
    libeccodes0 \
    libeccodes-dev \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the application code
COPY . /app/

# Expose port (if using HTTP transport)
EXPOSE 8050

# Default command to run the MCP server, example using HTTP transport
CMD ["python", "src/meteo_swiss_mcp_server.py", "--transport=streamable-http", "--host=0.0.0.0", "--port=8050"]