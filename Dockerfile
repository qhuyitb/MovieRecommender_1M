FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy project
COPY . /app

# Expose Streamlit port
EXPOSE 8501

ENV STREAMLIT_SERVER_HEADLESS=true

# Run streamlit
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
