import wandb
import os
import time
from captcha.model import Resnet18
import torch
from omegaconf import OmegaConf


def load_model(artifact, cfg):
    logdir = "models"
    model_checkpoint = os.environ.get("WANDB_ENTITY") + "/" + os.environ.get("WANDB_PROJECT") + "/captcha_model:latest"

    api = wandb.Api(
        api_key=os.getenv("WANDB_API_KEY"),
        overrides={"entity": os.getenv("WANDB_ENTITY"), "project": os.getenv("WANDB_PROJECT")},
    )
    artifact = api.artifact(model_checkpoint)
    artifact.download(root=logdir)
    file_name = artifact.files()[0].name
    print(f"Model downloaded to: {logdir}/{file_name}")
    return Resnet18.load_from_checkpoint(f"{logdir}/{file_name}", cfg.optimizer.Adam_opt)


def load_model_new(artifact):
    logdir = "models"
    model_checkpoint = f"{os.getenv('WANDB_ENTITY')}/{os.getenv('WANDB_PROJECT')}/captcha_model:latest"

    # Initialize Weights & Biases API
    api = wandb.Api(
        api_key=os.getenv("WANDB_API_KEY"),
        overrides={"entity": os.getenv("WANDB_ENTITY"), "project": os.getenv("WANDB_PROJECT")},
    )
    artifact = api.artifact(model_checkpoint)
    artifact.download(root=logdir)

    # Extract file name
    file_name = artifact.files()[0].name
    print(f"Model downloaded to: {logdir}/{file_name}")

    # Load the model
    checkpoint_path = os.path.join(logdir, file_name)
    model = Resnet18.load_from_checkpoint(checkpoint_path)
    model.eval()  # Set to evaluation mode

    return model


def test_model_speed():
    # Default configuration
    cfg = OmegaConf.create(
        {
            "dummy_data": False,
            "defaults": [{"model": "model"}, {"optimizer": "Adam_opt"}],
            "optimizer": {"Adam_opt": {"_target_": "torch.optim.Adam", "lr": 1e-3}},
            "model": {"hyperparameters": {"batch_size": 2, "epochs": 1}},
        }
    )
    model = load_model(os.getenv("MODEL_NAME"), cfg)
    # model = torch.load("models/model.pth")
    start = time.time()
    for i in range(100):
        print(f"Prediction {i}.")
        model(torch.rand(1, 1, 28, 28))
    end = time.time()
    assert end - start < 1
