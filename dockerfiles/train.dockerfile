# Base image
FROM python:3.11-slim AS base

# Install Python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copying the essential parts of the program
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY models/ models/
COPY reports/ reports/
COPY configs/ configs/

# Set the working directory and install dependencies
WORKDIR /
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
RUN pip install . --no-deps --no-cache-dir

# Which file should be executed
ENTRYPOINT ["python", "-u", "src/captcha/train.py"]

# Run the docker file with the following command assuming you have a .env file with your WANDB API key
#docker run --env-file .env --name <experiment_name> train:latest
