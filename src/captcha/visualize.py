import torch
import matplotlib.pyplot as plt
import random
from dataset import CaptchaDataset


# Assuming load_data is defined in the same script or imported
def visualize_samples(dataset, num_samples: int=5, title: str="Dataset Samples"):
    """
    Visualizes random samples from the given dataset.

    Parameters:
        dataset (torch.utils.data.TensorDataset): The dataset to visualize samples from.
        num_samples (int): The number of samples to visualize.
        title (str): Title for the visualization plot.
    """
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    fig.suptitle(title, fontsize=16)

    for i in range(num_samples):
        idx = random.randint(0, len(dataset) - 1)
        image, label = dataset[idx]

        if isinstance(image, torch.Tensor):
            image = image.squeeze().numpy()

        axes[i].imshow(image, cmap="gray")
        axes[i].set_title(f"Label: {label.item()}")
        axes[i].axis("off")

    plt.show()

def main():
    train_set = CaptchaDataset(processed_data_path="data/processed", data_type="train")
    test_set = CaptchaDataset(processed_data_path="data/processed", data_type="test")
    validation_set = CaptchaDataset(processed_data_path="data/processed", data_type="validation")

    # Visualize random samples from the train, validation, and test datasets
    visualize_samples(train_set, num_samples=5, title="Train Dataset Samples")
    visualize_samples(validation_set, num_samples=5, title="Validation Dataset Samples")
    visualize_samples(test_set, num_samples=5, title="Test Dataset Samples")

if __name__ == "__main__":
    main()
