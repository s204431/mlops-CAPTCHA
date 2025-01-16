from src.captcha.model import Resnet18
import torch

# 64% coverage


# Test the optimizer configuration
def test_optimizer_configuration():
    cfg = {"_target_": "torch.optim.Adam", "lr": 0.001}
    model = Resnet18(cfg)
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.defaults["lr"] == 0.001


# Test the forward pass
def test_forward_pass():
    cfg = {"_target_": "torch.optim.Adam", "lr": 0.001}
    model = Resnet18(cfg)
    input_tensor = torch.rand((1, 1, 224, 224))  # Batch size of 1, 1 channel, 224x224 image
    output = model(input_tensor)
    assert output.shape == (1, 20)  # 20 classes


# Test the training step
def test_training_step():
    cfg = {"_target_": "torch.optim.Adam", "lr": 0.001}
    model = Resnet18(cfg)
    batch = (torch.rand((32, 1, 224, 224)), torch.randint(0, 20, (32,)))  # Batch of 32
    loss = model.training_step(batch)
    assert isinstance(loss, torch.Tensor)


# Test for validation step
def test_validation_step():
    cfg = {"_target_": "torch.optim.Adam", "lr": 0.001}
    model = Resnet18(cfg)
    batch = (torch.rand((32, 1, 224, 224)), torch.randint(0, 20, (32,)))
    model.validation_step(batch)  # Should not raise exceptions


# Test for different batch sizes
def test_batch_size():
    cfg = {"_target_": "torch.optim.Adam", "lr": 0.001}
    model = Resnet18(cfg)
    for batch_size in [1, 16, 64]:
        input_tensor = torch.rand((batch_size, 1, 224, 224))
        output = model(input_tensor)
        assert output.shape == (batch_size, 20)
