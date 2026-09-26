FROM python:3.12-slim

# system deps: tesseract (+english), poppler (pdf2image), libgl (opencv)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# HF Spaces expects port 7860; Render injects $PORT
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-7860} --workers 2 --timeout 300 app.web:app"]
