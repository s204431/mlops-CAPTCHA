import random
from locust import HttpUser, between, task
import numpy as np
from PIL import Image
from pathlib import Path


def normalize(image: np.array) -> np.array:
    """
    Normalize an image by subtracting the mean and dividing by the standard deviation.
    Args:
        images (np.array): Image.
    Returns:
        np.array: Normalized image.
    """
    return (image - image.mean()) / image.std()


class User(HttpUser):
    """Locust user"""

    wait_time = between(1, 2)

    @task()
    def run_on_backend(self) -> None:
        """This tests that the backend service is responding with the expected output."""
        img_files = list(Path("tests/test_images/").glob("**/*.png"))
        random.shuffle(img_files)
        image = Image.open(img_files[0])
        image = np.array(image)
        image = np.expand_dims(image, 2)
        image = np.transpose(image, (2, 0, 1))  # Change to CHW format
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image = normalize(image)
        self.client.post("/predict", json={"image": image.tolist()})
