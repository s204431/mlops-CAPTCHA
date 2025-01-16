FROM python:3.11-slim
WORKDIR /bento
COPY src/captcha/bentoml_service.py .
COPY models/onnx_model.onnx .
COPY requirements_backend.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements_backend.txt
ENV PORT=3000
CMD bentoml serve bentoml_service:CaptchaClassifierService --port=$PORT
