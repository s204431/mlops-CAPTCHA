import numpy as np


def normalize(image: np.array) -> np.array:
    """
    Normalize an image by subtracting the mean and dividing by the standard deviation.
    Args:
        images (np.array): Image.
    Returns:
        np.array: Normalized image.
    """
    return (image - image.mean()) / image.std()
