# Multi-runtime container: Python + .NET + nginx
# For Azure Container Apps deployment

FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS base

# Install Python 3.11, nginx, and utilities
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    nginx \
    supervisor \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python

# Set up working directory
WORKDIR /app

# Copy Python inference service and scripts
COPY src/inference/ /app/inference/
COPY scripts/ /app/scripts/
COPY models/ /app/models/
COPY version.json /app/version.json

# Install Python dependencies
RUN python -m pip install --no-cache-dir --break-system-packages -r /app/inference/requirements.txt

# Build .NET API
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src/ChessFEN.Api
COPY web/ChessFEN.Api/ChessFEN.Api/*.csproj ./
RUN dotnet restore
COPY web/ChessFEN.Api/ChessFEN.Api/ ./
RUN dotnet publish -c Release -o /app/publish

# Final image
FROM base AS final

# Copy .NET API
COPY --from=build /app/publish /app/api/

# Copy static web client
COPY web/ChessFEN.Client/ /app/wwwroot/

# Copy nginx config
COPY deploy/nginx.conf /etc/nginx/nginx.conf

# Copy supervisor config
COPY deploy/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Copy and setup startup script
COPY deploy/start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Create required directories
RUN mkdir -p /var/log/supervisor /var/run

# Expose port 80
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost/health || exit 1

# Start supervisor (manages nginx, python, dotnet)
CMD ["/app/start.sh"]
