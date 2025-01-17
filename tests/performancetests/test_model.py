# def load_model(artifact):
#    logdir = "models"
#    model_checkpoint = artifact
#    api = wandb.Api(
#        api_key=os.getenv("WANDB_API_KEY"),
#        overrides={"entity": os.getenv("WANDB_ENTITY"), "project": os.getenv("WANDB_PROJECT")},
#    )
#    artifact = api.artifact(f"{model_checkpoint}:latest")
#   #artifact = api.use_model(os.getenv("MODEL_NAME"))
#    artifact.download(root=logdir)
#    file_name = artifact.files()[0].name
#    return Resnet18.load_from_checkpoint(f"{logdir}/{file_name}")


# def test_model_speed():
#    model = load_model(os.getenv("MODEL_NAME"))
#    start = time.time()
#    for _ in range(100):
#        model(torch.rand(1, 1, 28, 28))
#    end = time.time()
#    assert end - start < 1


def test_test():
    assert 1 == 1
