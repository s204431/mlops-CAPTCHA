import torch
#from captcha.dataloader import load_data
from captcha.model import MyAwesomeModel
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision import transforms
import torchvision.datasets as datasets
from captcha import _ROOT

def evaluate():
    #_, _, test_set = load_data()
    _, _, test_set = load_dummy()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)
    model = MyAwesomeModel()  # this is our LightningModule
    model_checkpoint = f"{_ROOT}/models/{model.pth}"
    model.load_state_dict(torch.load(model_checkpoint))

    trainer = Trainer()  # this is our Trainer
    trainer.test(model, test_dataloader)


def load_dummy(): #Temporary function with dummy data
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
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    evaluate()