import bentoml
import numpy as np
from PIL import Image


def normalize(image: np.array) -> np.array:
    """
    Normalize an image by subtracting the mean and dividing by the standard deviation.
    Args:
        images (np.array): Image.
    Returns:
        np.array: Normalized image.
    """
    return (image - image.mean()) / image.std()


if __name__ == "__main__":
    image = Image.open("data/raw/3_62249.png")
    # image = image.resize((53, 32))  # Resize to match the minimum input size of the model
    image = np.array(image)
    image = np.expand_dims(image, 2)
    image = np.transpose(image, (2, 0, 1))  # Change to CHW format
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = normalize(image)

    with bentoml.SyncHTTPClient("https://backend-1048604560911.europe-west1.run.app") as client:
        resp = client.predict(image=image)
        print(resp)
