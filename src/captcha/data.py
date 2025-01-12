import os
import random
from pathlib import Path
from zipfile import ZipFile

import torch
import typer
import gdown
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset, TensorDataset
import torchvision.transforms as transforms
from tqdm import tqdm
from captcha import _ROOT
from torch.profiler import profile, ProfilerActivity# didnt work for me ->, tensorboard_trace_handler

RAW_DATA_PATH = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed")


class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, raw_data_path: Path) -> None:
        """
        Initialize the dataset with the given path to raw data.
        """
        self.data_path = raw_data_path
        self._png_files = list(self.data_path.glob("**/*.png"))

    def __len__(self) -> int:
        """
        Return the length of the dataset.
        (Implement logic for counting images or data samples, if needed.)
        """
        return len(self._png_files)

    def __getitem__(self, index: int):
        """
        Return the sample (image, label, etc.) at the given index.
        (Implementation depends on your downstream use-case.)
        """
        img_path = self._png_files[index]
        with Image.open(img_path) as img:
            return img
        
    def preprocess(self, output_folder: Path, subset_size: int = 10000) -> None:

        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            output_folder.mkdir(parents=True, exist_ok=True)

            img_files = list(self.data_path.glob("**/*.png"))
            random.shuffle(img_files)
            img_files = img_files[:min(subset_size, len(img_files))]

            # Extracting labels 
            all_labels = []
            for img_path in img_files:
                label_str = img_path.stem.split('_')[0]
                all_labels.append(label_str)
            
            # Convert to a sorted list of unique labels
            class_names = sorted(list(set(all_labels)))
            # Create a dictionary {label_str: class_idx}
            label_to_idx = {lbl: i for i, lbl in enumerate(class_names)}
            
            # Split into train/val/test
            total_count = len(img_files)
            test_count = int(0.10 * total_count)
            val_count = int(0.20 * total_count)
            train_count = total_count - test_count - val_count

            train_files = img_files[:train_count]
            val_files = img_files[train_count : train_count + val_count]
            test_files = img_files[train_count + val_count :]

            transform = transforms.Compose([
                transforms.ToTensor(),
            ])

            # Return a tuple (image_tensor, label_int)
            def process_split(image_paths):
                images, labels = [], []
                for img_path in image_paths:
                    label_str = img_path.stem.split('_')[0]
                    label_int = label_to_idx[label_str]
                    with Image.open(img_path) as img:
                        img_tensor = transform(img)
                        images.append(img_tensor)
                        labels.append(label_int)
                images = torch.stack(images) # Stack to (N, C, H, W)
                labels = torch.tensor(labels)  # Labels to tensor
                return images, labels


            # Split into datasets tensors 
            logger.info(f"\033[36mProcessing {train_count} images for train split...")
            train_images, train_labels = process_split(train_files)
            logger.info(f"\033[36mTrain images shape {train_images.shape}")
            torch.save(train_images, output_folder / "train_images.pt")
            torch.save(train_labels, output_folder / "train_labels.pt")

            logger.info(f"\033[36mProcessing {val_count} images for val split...")
            val_images, val_labels = process_split(val_files)
            logger.info(f"\033[36mVal images shape {val_images.shape}")
            torch.save(val_images, output_folder / "val_images.pt")
            torch.save(val_labels, output_folder / "val_labels.pt")

            logger.info(f"\033[36mProcessing {test_count} images for test split...")
            test_images, test_labels = process_split(test_files)
            logger.info(f"\033[36mTest images shape {test_images.shape}")
            torch.save(test_images, output_folder / "test_images.pt")
            torch.save(test_labels, output_folder / "test_labels.pt")

            # Save class names
            torch.save(class_names, output_folder / "class_names.pt")


            # Summary
            logger.info("\033[36mPreprocessing complete.")
            logger.info(f"\033[36mSplit summary:")
            logger.info(f"\033[36m  Train: {train_count} samples")
            logger.info(f"\033[36m  Val:   {val_count} samples")
            logger.info(f"\033[36m  Test:  {test_count} samples")
            logger.info(f"\033[36mFound {len(class_names)} unique classes: {class_names}")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))


def normalize(images: torch.Tensor) -> torch.Tensor:
    """
    Normalize a batch of images by subtracting the mean and dividing by the standard deviation.
    Args:
        images (torch.Tensor): Batch of images.
    Returns:
        torch.Tensor: Normalized images.
    """
    return (images - images.mean()) / images.std()

def download_extract_dataset(raw_data_path: Path, zip_url: str) -> None:

    # If folder is not empty, skip download
    if raw_data_path.exists() and any(raw_data_path.iterdir()):
        logger.info(f"'{raw_data_path}' is not empty. Skipping download & extraction...")
        return
    
    logger.info(f"Downloading dataset from Google Drive to {zip_url}...")
    raw_data_path.mkdir(parents=True, exist_ok=True)

    zip_path = raw_data_path / "dataset.zip"
    gdown.download(zip_url, str(zip_path), quiet=False)

    # extract the dataset.zip file 
    logger.info("Extracting dataset...")
    with ZipFile(zip_path, 'r') as zip_ref:
        for file in tqdm(zip_ref.namelist(), desc="Extracting files", unit="file"):
            zip_ref.extract(member=file, path=raw_data_path)

    # Remove the zip file after excectracting
    zip_path.unlink()
    logger.success("\033[32m✅ Dataset extracted.")

    # Remove the extracted `Dataset` folder
    extracted_folder = raw_data_path / "Dataset"
    if extracted_folder.exists() and extracted_folder.is_dir():
        for item in extracted_folder.iterdir():
            target_path = raw_data_path / item.name
            item.replace(target_path)
        extracted_folder.rmdir()
    logger.info("\033[36mFiles moved and `Dataset` folder deleted.")

    for file in list(raw_data_path.glob("**/*.png")):
        if file.parent != raw_data_path:
            new_path = raw_data_path / file.name
            file.replace(new_path)

    logger.info(f"\033[36mAll PNG files moves to {RAW_DATA_PATH}")

def preprocess():

    zip_url = "https://drive.google.com/uc?id=1HyOhjM2WgmRucD-czc3UzTaFBAtx7-ae"
    download_extract_dataset(RAW_DATA_PATH, zip_url=zip_url)

    logger.info("\033[36mPreprocessing data...")
    dataset = MyDataset(RAW_DATA_PATH)
    dataset.preprocess(PROCESSED_DATA_PATH, subset_size=10000)
    logger.success("\033[32m ✅Data preprocessing complete.")

def load_data() -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train and test datasets for CAPTCHA data set."""
    path = f"{_ROOT}/data/processed/"

    train_images = torch.load(f"{path}/train_images.pt")
    train_target = torch.load(f"{path}/train_labels.pt")
    test_images = torch.load(f"{path}/val_images.pt")
    test_target = torch.load(f"{path}/val_labels.pt")
    test_images = torch.load(f"{path}/test_images.pt")
    test_target = torch.load(f"{path}/test_labels.pt")


    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    validation_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)
    return train_set, validation_set, test_set


def main():
    typer.run(preprocess)

if __name__ == "__main__":
    main()

