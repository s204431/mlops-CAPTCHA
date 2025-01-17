import typer
import os
import wandb
from typing import List


def link_model(
    artifact_path: str,
    aliases: List[str] = typer.Option(["staging"], "--alias", "-a", help="Aliases to link the artifact with."),
) -> None:
    """
    Stage a specific model to the model registry.

    Args:
        artifact_path: Path to the artifact to stage.
            Should be of the format "entity/project/artifact_name:version".
        aliases: List of aliases to link the artifact with.

    Example:
        python link_model.py MLOps-Captcha/Captcha/CaptchaModel:V0 --alias staging --alias production
    """
    if artifact_path == "":
        typer.echo("No artifact path provided. Exiting.")
        return

    api = wandb.Api(
        api_key=os.getenv("WANDB_API_KEY"),
        overrides={
            "entity": os.getenv("WANDB_ENTITY"),
            "project": os.getenv("WANDB_PROJECT"),
        },
    )
    _, _, artifact_name_version = artifact_path.split("/")
    artifact_name, _ = artifact_name_version.split(":")

    artifact = api.artifact(artifact_path)
    artifact.link(
        # target_path=f"{os.getenv('WANDB_ENTITY')}/model-registry/{artifact_name}",
        target_path=f"{os.getenv('WANDB_ENTITY')}/model-registry/{artifact_name}",
        aliases=aliases,
    )
    artifact.save()
    typer.echo(f"Artifact {artifact_path} linked to aliases: {aliases}")


def main():
    typer.run(link_model)


if __name__ == "__main__":
    main()
