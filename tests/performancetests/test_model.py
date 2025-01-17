import wandb
import os
import time
from captcha.model import Resnet18
import torch


def load_model(artifact):
    logdir = "models"
    model_checkpoint = os.getenv("MODEL_NAME")

    api = wandb.Api(
        api_key=os.getenv("WANDB_API_KEY"),
        overrides={"entity": os.getenv("WANDB_ENTITY"), "project": os.getenv("WANDB_PROJECT")},
    )
    artifact = api.artifact(model_checkpoint)
    artifact.download(root=logdir)
    file_name = artifact.files()[0].name
    return Resnet18.load_from_checkpoint(f"{logdir}/{file_name}")


def test_model_speed():
    model = load_model(os.getenv("MODEL_NAME"))
    start = time.time()
    for _ in range(100):
        model(torch.rand(1, 1, 28, 28))
    end = time.time()
    assert end - start < 1
