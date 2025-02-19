import torch
import matplotlib.pyplot as plt
import io
import numpy as np

from google.cloud import storage
from torch.utils.data import Dataset
from pathlib import Path

STATE = "remote"


def show_image_and_target(images: torch.Tensor, target: np.array, show: bool = True) -> None:
    """Show images and target labels."""
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].squeeze(), cmap="gray")
        ax.set_title(f"Label: {target[i]}")
        ax.axis("off")
    if show:
        plt.show()


def dataset_statistics(datadir: str = "data/processed") -> None:
    train_dataset = CaptchaDataset(datadir, "train", state=STATE)
    val_dataset = CaptchaDataset(datadir, "validation", state=STATE)
    test_dataset = CaptchaDataset(datadir, "test", state=STATE)
    class_names = np.array(torch.load(f"{datadir}/class_names.pt"))

    print("Train dataset:")
    print(f"Number of images: {len(train_dataset)}")
    print(f"Image shape: {train_dataset[0][0].shape}")
    print("\n")
    print("Validation dataset:")
    print(f"Number of images: {len(val_dataset)}")
    print(f"Image shape: {val_dataset[0][0].shape}")
    print("\n")
    print("Test dataset:")
    print(f"Number of images: {len(test_dataset)}")
    print(f"Image shape: {test_dataset[0][0].shape}")
    print("\n")
    print(f"Class names: {class_names}")

    show_image_and_target(train_dataset.images[:25], class_names[train_dataset.target[:25].data.numpy()], show=False)
    plt.savefig("captcha_images.png")
    plt.close()

    train_label_distribution = torch.bincount(train_dataset.target)
    val_label_distribution = torch.bincount(val_dataset.target)
    test_label_distribution = torch.bincount(test_dataset.target)

    plt.bar(class_names, train_label_distribution)
    plt.title("Train label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig("train_label_distribution.png")
    plt.close()

    plt.bar(class_names, val_label_distribution)
    plt.title("Validation label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig("val_label_distribution.png")
    plt.close()

    plt.bar(class_names, test_label_distribution)
    plt.title("Test label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig("test_label_distribution.png")
    plt.close()


class CaptchaDataset(Dataset):
    """Custom dataset for the CAPTCHA data."""

    def __init__(self, processed_data_path: Path, data_type: str, state: str) -> None:
        """
        Initialize the dataset with the given path to processed data.
        """
        self.data_path = processed_data_path
        self.data_type = data_type
        self.state = state
        self.load_data()

    def load_data(self) -> None:
        """Return train, validation and test datasets for CAPTCHA data set."""
        if self.state == "local":
            if self.data_type == "train":
                self.images = torch.load(f"{self.data_path}/train_images.pt")
                self.target = torch.load(f"{self.data_path}/train_labels.pt")
            elif self.data_type == "validation" or self.data_type == "val":
                self.images = torch.load(f"{self.data_path}/val_images.pt")
                self.target = torch.load(f"{self.data_path}/val_labels.pt")
            else:
                self.images = torch.load(f"{self.data_path}/test_images.pt")
                self.target = torch.load(f"{self.data_path}/test_labels.pt")
        elif self.state == "remote":
            client = storage.Client()
            bucket = client.get_bucket("mlops_captcha_bucket")
            base_path = "data/processed"  # Base path in GCS bucket

            if self.data_type == "train":
                image_blob = bucket.blob(f"{base_path}/train_images.pt")
                target_blob = bucket.blob(f"{base_path}/train_labels.pt")
            elif self.data_type == "validation" or self.data_type == "val":
                image_blob = bucket.blob(f"{base_path}/val_images.pt")
                target_blob = bucket.blob(f"{base_path}/val_labels.pt")
            else:
                image_blob = bucket.blob(f"{base_path}/test_images.pt")
                target_blob = bucket.blob(f"{base_path}/test_labels.pt")

            self.images = torch.load(io.BytesIO(image_blob.download_as_bytes()))
            self.target = torch.load(io.BytesIO(target_blob.download_as_bytes()))

        return self.images, self.target

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return image and target tensor."""
        return self.images[idx], self.target[idx]

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return self.images.shape[0]


def main():
    dataset_statistics()


if __name__ == "__main__":
    main()
