import pytorch_lightning as pl
import torch
from torch import nn
import timm
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision import transforms
import torchvision.datasets as datasets

class MyAwesomeModel(pl.LightningModule):
    """My awesome model."""

    def __init__(self) -> None:
        super().__init__()

        # Create the ResNet18 model
        self.model = timm.create_model('resnet18', pretrained=True, in_chans=1)

        # Freeze all parameters in the pretrained layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace the final fully connected layer with your custom layer
        num_features = self.model.fc.in_features                 
        num_classes = 10                                        
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
        print("train_loss", loss) #REPLACE WITH WEIGHTB
        return loss

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=1e-3)

if __name__ == "__main__":
    # Initialize the model
    model = MyAwesomeModel()

    # Print model architecture
    print(f"Model architecture: {model}")

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
    trainer = pl.Trainer(max_epochs=10)

    # Train the model
    trainer.fit(model, train_loader, val_loader)




    #print(f"Model architecture: {model}")
    #print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    #dummy_input = torch.randn(1, 3, 224, 224)
    #output = model(dummy_input)
    #print(f"Output shape: {output.shape}")

    #for name, param in model.named_parameters():
    #    print(f"{name}: {'Trainable' if param.requires_grad else 'Frozen'}")