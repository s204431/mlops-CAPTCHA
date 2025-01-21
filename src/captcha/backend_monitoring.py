import json
import os
from pathlib import Path

import anyio
import nltk
import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from google.cloud import storage
import numpy as np
import torch

nltk.download("words")
nltk.download("wordnet")
nltk.download("omw-1.4")

MONITORING_BUCKET = "mlops_captcha_monitoring"
DATA_BUCKET = "mlops_captcha_bucket"
CLASSES = np.array(["2", "3", "4", "5", "6", "7", "8", "a", "b", "c", "d", "e", "f", "g", "m", "n", "p", "w", "x", "y"])


def normalize(image: np.array) -> np.array:
    """
    Normalize an image by subtracting the mean and dividing by the standard deviation.
    Args:
        images (np.array): Image.
    Returns:
        np.array: Normalized image.
    """
    return (image - image.mean()) / image.std()


def extract_features(images):
    """Extract basic image features from a set of images."""
    features = []
    for img in images:
        img = img.squeeze(0)
        min_pixel = np.min(img)
        max_pixel = np.max(img)
        sharpness = np.mean(np.abs(np.gradient(img)))
        features.append([min_pixel, max_pixel, sharpness])
    return np.array(features)


def run_analysis(reference_data: pd.DataFrame, current_data: pd.DataFrame) -> None:
    """Run the analysis and return the report."""
    text_overview_report = Report(metrics=[DataDriftPreset()])
    text_overview_report.run(reference_data=reference_data, current_data=current_data)
    text_overview_report.save_html("monitoring.html")


def load_data():
    if not os.path.exists("train_images.pt") or not os.path.exists("train_labels.pt"):
        bucket = storage.Client.create_anonymous_client().bucket(DATA_BUCKET)
        blob_images = bucket.blob("data/processed/train_images.pt")
        blob_targets = bucket.blob("data/processed/train_labels.pt")
        blob_images.download_to_filename(os.path.basename(blob_images.name))
        blob_targets.download_to_filename(os.path.basename(blob_targets.name))
    images = torch.load("train_images.pt").data.numpy()
    for i in range(images.shape[0]):
        images[i] = normalize(images[i])
    targets = torch.load("train_labels.pt").data.numpy()
    img_features = extract_features(images)
    combined_data = np.column_stack((img_features, targets))
    df = pd.DataFrame(combined_data, columns=["Min Pixel", "Max Pixel", "Sharpness", "target"])
    return df


def lifespan(app: FastAPI):
    """Load the data and class names before the application starts."""
    global training_data

    training_data = load_data()

    yield

    del training_data


app = FastAPI(lifespan=lifespan)


def load_latest_files(directory: Path, n: int) -> pd.DataFrame:
    """Load the N latest prediction files from the directory."""
    # Download the latest prediction files from the GCP bucket
    download_files(n=n)

    # Get all prediction files in the directory
    files = directory.glob("prediction_*.json")

    # Sort files based on when they where created
    files = sorted(files, key=os.path.getmtime)

    # Get the N latest files
    latest_files = files[-n:]

    # Load or process the files as needed
    images, targets = [], []
    for file in latest_files:
        with file.open() as f:
            data = json.load(f)
            image = data["image"]
            image = np.array(image).reshape(1, 52, 32)
            images.append(image)
            targets.append(np.where(CLASSES == data["prediction"])[0])
    images = np.stack(images)
    img_features = extract_features(images)
    targets = np.array(targets)
    combined_data = np.column_stack((img_features, targets))
    df = pd.DataFrame(combined_data, columns=["Min Pixel", "Max Pixel", "Sharpness", "target"])
    return df


def download_files(n: int = 5) -> None:
    """Download the N latest prediction files from the GCP bucket."""
    bucket = storage.Client.create_anonymous_client().bucket(MONITORING_BUCKET)
    blobs = bucket.list_blobs(prefix="prediction_")
    blobs = sorted(blobs, key=lambda x: x.updated, reverse=True)
    latest_blobs = blobs[:n]

    for blob in latest_blobs:
        blob.download_to_filename(blob.name.replace(":", "_").replace(" ", "_"))


@app.get("/report", response_class=HTMLResponse)
async def get_report(n: int = 5):
    """Generate and return the report."""
    prediction_data = load_latest_files(Path("."), n=n)
    run_analysis(training_data, prediction_data)
    async with await anyio.open_file("monitoring.html", encoding="utf-8") as f:
        html_content = await f.read()
    return HTMLResponse(content=html_content, status_code=200)
