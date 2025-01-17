import bentoml
import numpy as np
from PIL import Image
from pathlib import Path
import requests


def normalize(image: np.array) -> np.array:
    """
    Normalize an image by subtracting the mean and dividing by the standard deviation.
    Args:
        images (np.array): Image.
    Returns:
        np.array: Normalized image.
    """
    return (image - image.mean()) / image.std()


def test_backend():
    """This tests that the backend service is responding with the expected output."""
    img_files = list(Path("tests/test_images/").glob("**/*.png"))
    for i in range(0, len(img_files)):
        image = Image.open(img_files[i])
        image = np.array(image)
        image = np.expand_dims(image, 2)
        image = np.transpose(image, (2, 0, 1))  # Change to CHW format
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image = normalize(image)
        with bentoml.SyncHTTPClient("https://backend-1048604560911.europe-west1.run.app") as client:
            resp = client.predict(image=image)
        assert resp is not None
        assert resp.shape == (1, 20)


def test_frontend():
    """Checks that the frontend is responding with a 200 status code."""
    response = requests.get("https://frontend-1048604560911.europe-west1.run.app")
    assert response.status_code == 200
