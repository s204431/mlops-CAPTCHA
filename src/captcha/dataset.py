import torch
from PIL import Image
from torch.utils.data import Dataset
from captcha import _ROOT
from pathlib import Path

class CaptchaDataset(Dataset):
    """Custom dataset for the CAPTCHA data."""

    def __init__(self, processed_data_path: Path, data_type: str) -> None:
        """
        Initialize the dataset with the given path to processed data.
        """
        self.data_path = processed_data_path
        self.data_type = data_type
        self.load_data()

    def load_data(self) -> None:
        """Return train, validation and test datasets for CAPTCHA data set."""

        if self.data_type == "train":
            self.images = torch.load(f"{self.data_path}/train_images.pt")
            self.target = torch.load(f"{self.data_path}/train_labels.pt")
        elif self.data_type == "validation" or self.data_type == "val":
            self.images = torch.load(f"{self.data_path}/val_images.pt")
            self.target = torch.load(f"{self.data_path}/val_labels.pt")
        else:
            self.images = torch.load(f"{self.data_path}/test_images.pt")
            self.target = torch.load(f"{self.data_path}/test_labels.pt")
        return self.images, self.target

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return image and target tensor."""
        return self.images[idx], self.target[idx]

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return self.images.shape[0]
