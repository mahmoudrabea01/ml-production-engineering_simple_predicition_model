# Use a lightweight Python 3.12 image.
FROM python:3.12-slim

# Install uv from its official container image.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory inside the container.
WORKDIR /app

# Prevent Python from buffering logs.
ENV PYTHONUNBUFFERED=1

# Install dependencies using the lockfile.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy application code and trained model.
COPY production_app/ ./production_app/
COPY model/ ./model/

# Document the port used by FastAPI.
EXPOSE 8000

# Run FastAPI, listening on all container interfaces.
CMD [ "uv" , "run" , "--no-sync" , "uvicorn" , "production_app.api:app" , "--host" , "0.0.0.0" , "--port" , "8000"]
