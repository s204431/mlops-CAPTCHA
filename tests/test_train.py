import tempfile
import torch.nn as nn
from unittest.mock import patch
from pathlib import Path
import torch
from captcha.train import train, load_dummy
from omegaconf import OmegaConf
from torch.utils.data import Dataset
import os

# 95% coverage


class DummyDataset(Dataset):
    """A minimal dataset implementation for testing."""

    def __init__(self, num_samples=10):
        self.data = torch.rand(num_samples, 1, 28, 28)  # Dummy image data
        self.labels = torch.randint(0, 10, (num_samples,))  # Dummy labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class DummyModel(nn.Module):
    """A lightweight dummy model for testing."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))


def test_train():
    os.environ["WANDB_MODE"] = "dryrun"
    # Create a temporary directory for testing outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock _ROOT to point to temporary directory
        with patch("captcha.train._ROOT", temp_dir):
            # Ensure models directory exists
            models_dir = Path(temp_dir) / "models"
            models_dir.mkdir(parents=True, exist_ok=True)

            # Replace CaptchaDataset with a dummy dataset
            with (
                patch("captcha.train.CaptchaDataset", side_effect=lambda *_: DummyDataset()),
                patch("captcha.train.Resnet18", return_value=DummyModel()),
                patch("captcha.train.pl.Trainer") as MockTrainer,
            ):
                # Mock Trainer
                trainer_instance = MockTrainer.return_value
                trainer_instance.fit.return_value = None
                trainer_instance.test.return_value = None

                # Config for testing
                cfg = OmegaConf.create(
                    {
                        "model": {"hyperparameters": {"batch_size": 2, "epochs": 1}},
                        "optimizer": {"Adam_opt": {"lr": 0.001}},
                    }
                )

                # Run the train function
                train(cfg)

                # Assertions
                trainer_instance.fit.assert_called_once()
                trainer_instance.test.assert_called_once()
                assert (models_dir / "model.pth").exists()


def test_load_dummy():
    train_dataset, val_dataset, test_dataset = load_dummy()
    assert len(train_dataset) > 0
    assert len(val_dataset) > 0
    assert len(test_dataset) > 0
