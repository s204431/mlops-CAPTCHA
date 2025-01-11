import pytorch_lightning as pl
import torch
from torch import nn
import timm
import wandb
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision import transforms
import torchvision.datasets as datasets
import hydra
from hydra.utils import instantiate
from loguru import logger

class Resnet18(pl.LightningModule):
    """My awesome model."""

    def __init__(self, optimimzer_cfg, num_classes: int = 20) -> None:
        super().__init__()

        self.optimizer_cfg = optimimzer_cfg

        # Create the ResNet18 model
        self.model = timm.create_model('resnet18', pretrained=True, in_chans=1)

        # Freeze all parameters in the pretrained layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace the final fully connected layer with your custom layer
        num_features = self.model.fc.in_features                                                        
        self.model.fc = nn.Linear(num_features, num_classes)

        # Ensure the new fc layer's parameters are trainable
        for param in self.model.fc.parameters():
            param.requires_grad = True

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)

    def training_step(self, batch):
        """Training step."""
        img, target = batch
        y_pred = self(img)
        loss = self.loss_fn(y_pred, target)
        acc = (target == y_pred.argmax(dim=-1)).float().mean()
        self.log('train_loss', loss, on_epoch=True)
        self.log('train_acc', acc, on_epoch=True)
        #print(f"Train loss: {loss}, Train accuracy: {acc}")
        return loss
    
    def validation_step(self, batch) -> None:
        img, target = batch
        y_pred = self(img)
        loss = self.loss_fn(y_pred, target)
        acc = (target == y_pred.argmax(dim=-1)).float().mean()
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_acc", acc, on_epoch=True)
        #print(f"Val loss: {loss}, Val accuracy: {acc}")

    def test_step(self, batch) -> None:
        img, target = batch
        y_pred = self(img)
        loss = self.loss_fn(y_pred, target)
        acc = (target == y_pred.argmax(dim=-1)).float().mean()
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        #print(f"Test loss: {loss}, Test accuracy: {acc}")

    def configure_optimizers(self):
        """Configure optimizer."""
        return hydra.utils.instantiate(self.optimizer_cfg, params=self.parameters())
    
    def on_train_epoch_end(self) -> None:
        train_loss = self.trainer.callback_metrics.get('train_loss')
        train_acc = self.trainer.callback_metrics.get('train_acc')
        if train_loss is not None and train_acc is not None:
            logger.info(f"\033[36m🔄 Epoch {self.trainer.current_epoch + 1}\033[0m Training - \033[33mLoss: {train_loss:.4f}\033[0m | \033[32mAccuracy: {train_acc:.4f}\033[0m")

    def on_validation_epoch_end(self) -> None:
        val_loss = self.trainer.callback_metrics.get('val_loss')
        val_acc = self.trainer.callback_metrics.get('val_acc')
        if val_loss is not None and val_acc is not None:
            logger.info(f"\033[36m✅ Epoch {self.trainer.current_epoch + 1}\033[0m Validation - \033[33mLoss: {val_loss:.4f}\033[0m | \033[32mAccuracy: {val_acc:.4f}\033[0m")

    def on_test_epoch_end(self) -> None:
        test_loss = self.trainer.callback_metrics.get('test_loss')
        test_acc = self.trainer.callback_metrics.get('test_acc')
        if test_loss is not None and test_acc is not None:
            logger.info(f"\033[36m🧪 Test Results\033[0m - \033[33mLoss: {test_loss:.4f}\033[0m | \033[32mAccuracy: {test_acc:.4f}\033[0m")

@hydra.main(config_path="../../configs", config_name="default_config")
def main(cfg):

    # Initialize the model
    model = Resnet18(cfg.optimizer.Adam_opt)

    # Dummy Dataset (Replace with your real dataset)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = datasets.FakeData(
        size=1000,
        image_size=(1, 224, 224),
        num_classes=10,
        transform=transform
    )
    
    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Initialize PyTorch Lightning Trainer
    trainer = pl.Trainer(max_epochs=10) #, logger=pl.loggers.WandbLogger(project="Captcha"))

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Print model architecture
    #print(f"Model architecture: {model}")

    #print(f"Model architecture: {model}")
    #print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    #dummy_input = torch.randn(1, 3, 224, 224)
    #output = model(dummy_input)
    #print(f"Output shape: {output.shape}")

    #for name, param in model.named_parameters():
    #    print(f"{name}: {'Trainable' if param.requires_grad else 'Frozen'}")

if __name__ == "__main__":
    main()