import random
import subprocess
from pathlib import Path
from zipfile import ZipFile

import torch
import typer
from loguru import logger
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm


RAW_DATA_PATH = Path("data/raw")  # Path where the raw dataset will be stored
PROCESSED_DATA_PATH = Path("data/processed")  # Path where the processed data will be saved


def push_data_to_dvc(raw_data_path: Path) -> bool:
    """Push processed data to DVC remote with correct file handling"""
    logger.info("Starting DVC push process...")

    try:
        # Add to DVC with verbose output
        logger.info("Running dvc add...")
        result = subprocess.run(["dvc", "add", str(raw_data_path), "-v"], check=True, capture_output=True, text=True)
        logger.info("DVC add complete: " + result.stdout)

        # Check for either data.dvc or data/processed.dvc
        dvc_file = Path("data.dvc")
        alt_dvc_file = Path(f"{raw_data_path}.dvc")

        # Check if either file exists
        if dvc_file.exists():
            actual_dvc_file = dvc_file
        # Check for alternative file
        elif alt_dvc_file.exists():
            actual_dvc_file = alt_dvc_file
        # No DVC file found
        else:
            logger.error("DVC add completed but no .dvc file was found")
            return False

        logger.info(f"Found DVC file at {actual_dvc_file}")

        # Add to git
        logger.info("Adding to git...")
        subprocess.run(["git", "add", str(actual_dvc_file)], check=True)

        # Commit changes
        try:
            subprocess.run(["git", "commit", "-m", f"Add {raw_data_path} to DVC", "--no-verify"], check=True)
            logger.info("Changes committed to git")
        except subprocess.CalledProcessError:
            logger.warning("Git commit failed - possibly nothing to commit")

        # Push to remote
        logger.info("Pushing to DVC remote...")
        subprocess.run(["dvc", "push", "--no-run-cache"], check=True, capture_output=True, text=True)
        logger.success("✅ Data pushed to DVC remote.")
        return True

    # Handle exceptions
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e.cmd}")
        if e.stdout:
            logger.error(f"Command output: {e.stdout}")
        if e.stderr:
            logger.error(f"Command error: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {type(e).__name__}: {str(e)}")
        return False


def download_extract_dataset_dvc(raw_data_path: Path) -> None:
    """Downloading the raw data from DVC remote (GCS)"""
    # Create raw data path if it does not exist
    raw_data_path.mkdir(parents=True, exist_ok=True)

    # Check if directory is empty
    is_empty = not raw_data_path.exists() or not any(raw_data_path.iterdir())

    # If not empty, skip download
    if not is_empty:
        logger.info(f"'{raw_data_path}' is not empty. Skipping download.")
        logger.success("✅ Using existing data.")
        return

    # Directory is empty - pull the raw data and extract
    logger.info("Directory is empty. Pulling and extracting data...")
    subprocess.run(["dvc", "pull", "--no-run-cache", str(raw_data_path)], check=True)
    logger.success("✅ Data pulled from GCS using DVC.")

    # Extract the dataset
    logger.info("Extracting dataset...")
    for file in tqdm(raw_data_path.glob("**/*.zip"), desc="Extracting files", unit="file"):
        with ZipFile(file, "r") as zip_ref:
            zip_ref.extractall(raw_data_path)
        # Remove the zip file
        file.unlink()

    # Move all files into data/raw folder and remove the folder "Dataset"
    dataset_folder = raw_data_path / "Dataset"
    if dataset_folder.exists() and dataset_folder.is_dir():
        for item in dataset_folder.iterdir():
            target_path = raw_data_path / item.name
            item.replace(target_path)
        dataset_folder.rmdir()
    logger.success("\033[32m✅ Dataset extracted.")


def preprocess_raw(input_folder: Path, output_folder: Path, subset_size: int = 10000) -> None:
    """Preprocess the dataset."""
    #    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    output_folder.mkdir(parents=True, exist_ok=True)

    # Get all image files and shuffle
    img_files = list(input_folder.glob("**/*.png"))
    random.shuffle(img_files)
    img_files = img_files[: min(subset_size, len(img_files))]

    # Extracting labels
    all_labels = []
    for img_path in img_files:
        label_str = img_path.stem.split("_")[0]
        all_labels.append(label_str)

    # Convert to a sorted list of unique labels
    class_names = sorted(list(set(all_labels)))
    # Create a dictionary {label_str: class_idx}
    label_to_idx = {lbl: i for i, lbl in enumerate(class_names)}

    # Split into train/val/test
    # Test 10%, Val 20%, Train 70%
    total_count = len(img_files)
    test_count = int(0.10 * total_count)
    val_count = int(0.20 * total_count)
    train_count = total_count - test_count - val_count

    # Split the files
    train_files = img_files[:train_count]
    val_files = img_files[train_count : train_count + val_count]
    test_files = img_files[train_count + val_count :]

    # Transform to tensor
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # Return a tuple (image_tensor, label_int)
    def process_split(image_paths):
        images, labels = [], []
        for img_path in image_paths:
            label_str = img_path.stem.split("_")[0]
            label_int = label_to_idx[label_str]
            with Image.open(img_path) as img:
                img_tensor = transform(img)
                images.append(img_tensor)
                labels.append(label_int)

        images = torch.stack(images)  # Stack to (N, C, H, W)
        images = normalize(images)  # Normalize
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
    logger.info("\033[36mSplit summary:")
    logger.info(f"\033[36m  Train: {train_count} samples")
    logger.info(f"\033[36m  Val:   {val_count} samples")
    logger.info(f"\033[36m  Test:  {test_count} samples")
    logger.info(f"\033[36mFound {len(class_names)} unique classes: {class_names}")


# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))


def normalize(images: torch.Tensor) -> torch.Tensor:
    """
    Normalize a batch of images by subtracting the mean and dividing by the standard deviation.
    Args:
        images (torch.Tensor): Batch of images.
    Returns:
        torch.Tensor: Normalized images.
    """
    return (images - images.mean()) / images.std()


def preprocess() -> None:
    """Preprocess the CAPTCHA dataset."""
    download_extract_dataset_dvc(RAW_DATA_PATH)

    # Preprocess the data
    logger.info("\033[36mPreprocessing data...")
    preprocess_raw(RAW_DATA_PATH, PROCESSED_DATA_PATH, subset_size=len(list(RAW_DATA_PATH.glob("**/*.png"))))
    logger.success("\033[32m ✅Data preprocessing complete.")

    # Push processed data to DVC and check success
    if push_data_to_dvc(PROCESSED_DATA_PATH):
        logger.success("\033[32m✅ Data preprocessing and DVC push complete.")
    else:
        logger.error("\033[31m❌ Data preprocessing complete but DVC push failed.")


def main():
    """Main function. Preprocesses the data."""
    typer.run(preprocess)


if __name__ == "__main__":
    main()
