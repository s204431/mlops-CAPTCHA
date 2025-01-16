import torch
import hydra
import os
import wandb
#from captcha.dataloader import load_data
from captcha.model import Resnet18
from captcha.dataset import CaptchaDataset
import pytorch_lightning as pl
from torch.utils.data import random_split
from torchvision import transforms
import torchvision.datasets as datasets
from captcha import _ROOT
from torch.profiler import profile, ProfilerActivity# didnt work for me ->, tensorboard_trace_handler
from captcha.logger import logger  # Import the configured Loguru logger
from omegaconf import DictConfig
from typing import Tuple
from dotenv import load_dotenv

def train(cfg: DictConfig) -> None:
    """Trains the model."""
    logger.info("\033[36m🚀 Starting training...")
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        data_path = f"{_ROOT}/data/processed/"
        train_set = CaptchaDataset(data_path, "train")
        validation_set = CaptchaDataset(data_path, "validation")
        test_set = CaptchaDataset(data_path, "test")
        #train_set, validation_set, test_set = load_dummy()
        train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=cfg.model.hyperparameters['batch_size'], shuffle=True, num_workers=4, persistent_workers=True)
        validation_dataloader = torch.utils.data.DataLoader(validation_set, batch_size=cfg.model.hyperparameters['batch_size'], num_workers=4, persistent_workers=True)
        test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=cfg.model.hyperparameters['batch_size'], num_workers=4, persistent_workers=True)
        model = Resnet18(cfg.optimizer.Adam_opt)  # this is our LightningModule
        #early_stopping_callback = EarlyStopping(monitor="val_loss", patience=3, verbose=True, mode="min")

        trainer = pl.Trainer(
            max_epochs=cfg.model.hyperparameters['epochs'],
            limit_train_batches=0.1,
            limit_val_batches=0.1,
            #callbacks=[early_stopping_callback],
            logger=pl.loggers.WandbLogger(project="Captcha"),
            enable_progress_bar=False
        )  # this is our Trainer

        trainer.fit(model, train_dataloader, validation_dataloader)
        logger.info("\033[36m🏁 Training completed. Starting testing...")
        trainer.test(model, test_dataloader)
        #trainer.test(model, test_dataloader)
        logger.info("\033✅ Testing completed.")
        torch.save(model.state_dict(), f"{_ROOT}/models/model.pth")
        # save the model to the outputs directory based on the time of run logged by hydra logger
        logger.info(f"\033[36m💾 Model saved to {_ROOT}/models/model.pth")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

def load_dummy() -> Tuple[datasets.FakeData, datasets.FakeData, datasets.FakeData]:
    """Loads a dummy dataset."""
    transform = transforms.Compose([
        transforms.Resize((52, 32)),
        transforms.ToTensor()
    ])

    dataset = datasets.FakeData(
        size=1000,
        image_size=(1, 52, 32),
        num_classes=10,
        transform=transform
    )
    
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