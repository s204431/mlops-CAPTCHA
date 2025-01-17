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
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch("captcha.train._ROOT", temp_dir):
            models_dir = Path(temp_dir) / "models"
            models_dir.mkdir(parents=True, exist_ok=True)

            with (
                patch("captcha.train.CaptchaDataset", side_effect=lambda *_: DummyDataset()),
                patch("captcha.train.Resnet18", return_value=DummyModel()),
                patch("captcha.train.pl.Trainer") as MockTrainer,
            ):
                trainer_instance = MockTrainer.return_value
                trainer_instance.fit.side_effect = lambda *args, **kwargs: print("Trainer.fit called")
                trainer_instance.test.side_effect = lambda *args, **kwargs: print("Trainer.test called")

                # Default configuration
                cfg = OmegaConf.create(
                    {
                        "dummy_data": True,
                        "defaults": [{"model": "model"}, {"optimizer": "Adam_opt"}],
                        "optimizer": {"Adam_opt": {"_target_": "torch.optim.Adam", "lr": 1e-3}},
                        "model": {"hyperparameters": {"batch_size": 2, "epochs": 1}},
                    }
                )

                # Run the training function
                train(cfg)

                # Assertions
                trainer_instance.fit.assert_called_once()
                trainer_instance.test.assert_called_once()
                print(f"Model saved to: {models_dir / 'model.pth'}")
                assert (models_dir / "model.pth").exists(), "Model was not saved as expected"


def test_load_dummy():
    train_dataset, val_dataset, test_dataset = load_dummy()
    assert len(train_dataset) > 0
    assert len(val_dataset) > 0
    assert len(test_dataset) > 0
