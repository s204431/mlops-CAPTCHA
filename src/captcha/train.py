import torch
import hydra
from pathlib import Path
import os
import subprocess
import wandb

# from captcha.dataloader import load_data
from captcha.model import Resnet18
from captcha.dataset import CaptchaDataset
import pytorch_lightning as pl
from torch.utils.data import random_split
from torchvision import transforms
import torchvision.datasets as datasets
from captcha import _ROOT
from captcha.logger import logger  # Import the configured Loguru logger
from omegaconf import DictConfig
from typing import Tuple
from dotenv import load_dotenv


def pull_data_from_dvc(data_path: str | Path) -> bool:
    """
    Check if processed data exists, if not pull from DVC.

    Args:
        data_path: Path to the processed data directory (can be string or Path)
    Returns:
        bool: True if data is available or successfully pulled
    """
    # Convert to Path object if string
    path = Path(data_path)

    logger.info("\033[36m📥 Checking for Processed data...")

    if path.exists() and any(path.iterdir()):
        # Verify essential files are present
        required_files = [
            "train_images.pt",
            "train_labels.pt",
            "val_images.pt",
            "val_labels.pt",
            "test_images.pt",
            "test_labels.pt",
            "class_names.pt",
        ]

        missing_files = [f for f in required_files if not (path / f).exists()]

        if not missing_files:
            logger.info("\033[36m📦 Found processed data.")
            return True

        logger.info("\033[36m📦 Some required files are missing.")

    else:
        logger.info("\033[36m📦 Processed data directory is empty or doesn't exist.")

    # Try to pull from DVC
    logger.info("\033[36m📦 Pulling processed data from DVC...")

    try:
        subprocess.run(["dvc", "pull", "--no-run-cache", str(path)], check=True)
        logger.success("\033[36m📦 Data pulled from DVC.")

        if path.exists() and any(path.iterdir()):
            logger.success("\033[36m📦 Successfully processed data.")
            return True
        else:
            logger.error("❌ DVC pull completed but data directory is still empty.")
            return False

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to pull data from DVC: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error while pulling data: {str(e)}")
        return False


def train(cfg: DictConfig) -> None:
    """Trains the model."""
    logger.info("\033[36m🚀 Starting training...")
    run = wandb.init(project="Captcha")

    if cfg.get("dummy_data", False):
        # Load dummy data
        logger.info("\033[36m📦 Using dummy dataset for training.")
        train_set, validation_set, test_set = load_dummy()
    else:
        # Check and pull data from DVC
        data_path = f"{_ROOT}/data/processed/"
        if not pull_data_from_dvc(data_path):
            logger.error("❌ Could not acquire processed data. Aborting training.")
            return
        train_set = CaptchaDataset(data_path, "train")
        validation_set = CaptchaDataset(data_path, "validation")
        test_set = CaptchaDataset(data_path, "test")

    train_dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=cfg.model.hyperparameters["batch_size"],
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )
    validation_dataloader = torch.utils.data.DataLoader(
        validation_set, batch_size=cfg.model.hyperparameters["batch_size"], num_workers=4, persistent_workers=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_set, batch_size=cfg.model.hyperparameters["batch_size"], num_workers=4, persistent_workers=True
    )

    model = Resnet18(cfg.optimizer.Adam_opt)  # LightningModule

    trainer = pl.Trainer(
        max_epochs=cfg.model.hyperparameters["epochs"],
        logger=pl.loggers.WandbLogger(project="Captcha"),
        enable_progress_bar=False,
    )  # Trainer
    trainer.fit(model, train_dataloader, validation_dataloader)
    logger.info("\033[36m🏁 Training completed. Starting testing...")
    trainer.test(model, test_dataloader)
    logger.info("\033✅ Testing completed.")
    torch.save(model.state_dict(), f"{_ROOT}/models/model.pth")
    logger.info(f"\033[36m💾 Model saved to {_ROOT}/models/model.pth")

    # Log model as an artifact
    final_test_acc = trainer.callback_metrics.get("test_acc", None)
    final_test_loss = trainer.callback_metrics.get("test_loss", None)
    print(final_test_acc, final_test_loss)
    artifact = wandb.Artifact(
        name="captcha_model",
        type="model",
        description="A model trained to predict captcha images",
        metadata={"test accuracy": final_test_acc, "test loss": final_test_loss},
    )
    artifact.add_file(f"{_ROOT}/models/model.pth")
    run.log_artifact(artifact)


def load_dummy() -> Tuple[datasets.FakeData, datasets.FakeData, datasets.FakeData]:
    """Loads a dummy dataset."""
    transform = transforms.Compose([transforms.Resize((52, 32)), transforms.ToTensor()])

    dataset = datasets.FakeData(size=1000, image_size=(1, 52, 32), num_classes=10, transform=transform)

    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    return train_dataset, val_dataset, test_dataset


@hydra.main(config_path=f"{_ROOT}/configs", config_name="default_config", version_base="1.1")
def main(cfg: DictConfig) -> None:
    """Main function. Simply runs the training."""
    load_dotenv()
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    train(cfg)


if __name__ == "__main__":
    main()
