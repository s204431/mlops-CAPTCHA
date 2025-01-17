import tempfile
from pathlib import Path
import torch
from src.captcha.dataset import CaptchaDataset, show_image_and_target
import numpy as np

# 37% coverage


def test_captcha_dataset():
    # Create temporary data directory with dummy data
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        temp_path.mkdir(parents=True, exist_ok=True)

        # Create dummy data
        train_images = torch.rand((100, 1, 28, 28))
        train_labels = torch.randint(0, 10, (100,))
        val_images = torch.rand((20, 1, 28, 28))
        val_labels = torch.randint(0, 10, (20,))
        test_images = torch.rand((10, 1, 28, 28))
        test_labels = torch.randint(0, 10, (10,))
        class_names = [str(i) for i in range(10)]

        torch.save(train_images, temp_path / "train_images.pt")
        torch.save(train_labels, temp_path / "train_labels.pt")
        torch.save(val_images, temp_path / "val_images.pt")
        torch.save(val_labels, temp_path / "val_labels.pt")
        torch.save(test_images, temp_path / "test_images.pt")
        torch.save(test_labels, temp_path / "test_labels.pt")
        torch.save(class_names, temp_path / "class_names.pt")

        # Test CaptchaDataset class
        dataset = CaptchaDataset(temp_path, "train")
        assert len(dataset) == 100
        assert dataset[0][0].shape == (1, 28, 28)
        assert dataset[0][1].item() in range(10)


def test_show_image_and_target():
    # Create dummy images and labels
    images = torch.rand((25, 1, 28, 28))
    target = np.random.randint(0, 10, 25)

    # Ensure it runs without error
    show_image_and_target(images, target, show=False)


# missing test for dataset_statistics
