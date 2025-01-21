FROM python:3.11-slim

WORKDIR /app

RUN pip install fastapi nltk evidently google-cloud-storage uvicorn pandas numpy torch --no-cache-dir

COPY src/captcha/backend_monitoring.py .

ENV PORT=8000

CMD exec uvicorn backend_monitoring:app --port $PORT --host 0.0.0.0 --workers 1
