import torch
import hydra
#from captcha.dataloader import load_data
from captcha.model import Resnet18
from captcha.data import load_data
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision import transforms
import torchvision.datasets as datasets
from captcha import _ROOT

def train(cfg):
    train_set, validation_set, test_set = load_data()
    #train_set, validation_set, test_set = load_dummy()
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=cfg.model.hyperparameters['batch_size'], shuffle=True)
    validation_dataloader = torch.utils.data.DataLoader(validation_set, batch_size=cfg.model.hyperparameters['batch_size'])
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=cfg.model.hyperparameters['batch_size'])
    model = Resnet18(cfg.optimizer.Adam_opt)  # this is our LightningModule
    #early_stopping_callback = EarlyStopping(monitor="val_loss", patience=3, verbose=True, mode="min")
    trainer = pl.Trainer(
        max_epochs=cfg.model.hyperparameters['epochs'],
        #limit_train_batches=0.2,
        #callbacks=[early_stopping_callback],
        logger=pl.loggers.WandbLogger(project="Captcha"),
    )  # this is our Trainer
    trainer.fit(model, train_dataloader, validation_dataloader)
    trainer.test(model, test_dataloader)
    #trainer.test(model, test_dataloader)
    torch.save(model.state_dict(), f"{_ROOT}/models/model.pth")


def load_dummy(): #Temporary function with dummy data
    # Dummy Dataset (Replace with your real dataset)
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

@hydra.main(config_path=f"{_ROOT}/configs", config_name="default_config")
def main(cfg):
    train(cfg)
    #train_set, validation_set, test_set = load_data()
    """train_set, validation_set, test_set = load_dummy()
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=cfg.model.hyperparameters['batch_size'], shuffle=True)
    validation_dataloader = torch.utils.data.DataLoader(validation_set, batch_size=cfg.model.hyperparameters['batch_size'])
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=cfg.model.hyperparameters['batch_size'])
    model = Resnet18(cfg.optimizer.Adam_opt)  # this is our LightningModule
    #early_stopping_callback = EarlyStopping(monitor="val_loss", patience=3, verbose=True, mode="min")
    trainer = Trainer(
        max_epochs=cfg.model.hyperparameters['epochs'],
        #limit_train_batches=0.2,
        #callbacks=[early_stopping_callback],
        #logger=loggers.WandbLogger(project="wandb_test"),
    )  # this is our Trainer
    trainer.fit(model, train_dataloader, validation_dataloader)
    #trainer.test(model, test_dataloader)"""

if __name__ == "__main__":
    main()