import torch
from captcha.model import Resnet18
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision import transforms
import torchvision.datasets as datasets
from captcha import _ROOT
from captcha.data import load_data
import hydra
from torch.profiler import profile, ProfilerActivity# didnt work for me ->, tensorboard_trace_handler
from omegaconf import DictConfig
from typing import Tuple

def evaluate(cfg: DictConfig) -> None:
    """Evaluates the model on the test set."""
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        _, _, test_set = load_data()
        #_, _, test_set = load_dummy()
        test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=cfg.model.hyperparameters['batch_size'])
        model = Resnet18(cfg.optimizer.Adam_opt)  # this is our LightningModule
        model_checkpoint = f"{_ROOT}/models/model.pth"
        model.load_state_dict(torch.load(model_checkpoint))

        trainer = Trainer()  # this is our Trainer
        trainer.test(model, test_dataloader)
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

@hydra.main(config_path=f"{_ROOT}/configs", config_name="default_config")
def main(cfg: DictConfig) -> None:
    """Main function. Evaluates the model on the test set."""
    evaluate(cfg)

if __name__ == "__main__":
    main()