FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN mkdir /app

WORKDIR /app

COPY requirements_frontend.txt /app/requirements_frontend.txt
COPY src/captcha/frontend.py /app/frontend.py

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements_frontend.txt
ENV PORT=8000
CMD streamlit run frontend.py --server.port $PORT
