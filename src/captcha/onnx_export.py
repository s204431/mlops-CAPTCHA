import torch
from hydra import compose, initialize
from captcha.model import Resnet18
from pathlib import Path
import os

with initialize(version_base=None, config_path="../../configs/optimizer"):
    cfg_opt = compose(config_name="Adam_opt")

model = Resnet18(cfg_opt)
path = Path(os.getcwd()).absolute()
model_checkpoint = f"{path}/models/model_fully_trained.pth"
model.load_state_dict(torch.load(model_checkpoint))
model.eval()
dummy_input = torch.randn(1, 1, 52, 32)
torch.onnx.export(
    model=model,
    args=(dummy_input,),
    f="onnx_model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)
