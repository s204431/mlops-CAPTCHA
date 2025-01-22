# Base image
FROM python:3.11-slim AS base


# Add build arg for WANDB
ARG WANDB_API_KEY
ENV WANDB_API_KEY=$WANDB_API_KEY
ENV WANDB_SILENT=true

# Install Python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Create necessary directories and set permissions
RUN mkdir -p /configs /models && \
    chmod 777 /models  # Ensure the models directory is writable


# Copying the essential parts of the program
COPY requirements.txt /requirements.txt
COPY pyproject.toml /pyproject.toml
COPY src/ /src/
COPY reports/ /reports/
COPY configs/ /configs/
COPY models/ /models/


# Set the working directory and install dependencies

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir



# Which file should be executed
ENTRYPOINT ["python", "-u", "src/captcha/train.py"]

# Run the docker file with the following command assuming you have a .env file with your WANDB API key
#docker run --env-file .env --name <experiment_name> train:latest
