from src.captcha.data import preprocess_raw, normalize
import torch
import tempfile
from pathlib import Path
from PIL import Image

# 71% coverage


# Test the preprocess_raw function on some sample data
def test_preprocess_raw_sample_data():
    # Create a temporary directory for input and output
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_folder = temp_path / "input"
        output_folder = temp_path / "output"
        input_folder.mkdir(parents=True, exist_ok=True)

        # Generate sample images with labels
        class_labels = ["cat", "dog", "bird"]
        num_samples = 12
        for i in range(num_samples):
            label = class_labels[i % len(class_labels)]
            img = Image.new("RGB", (64, 64), color=(i * 10, i * 20, i * 30))  # Create a colored image
            img.save(input_folder / f"{label}_{i}.png")

        # Call the preprocess_raw function
        preprocess_raw(input_folder, output_folder, subset_size=10)

        # Verify the output files
        assert (output_folder / "train_images.pt").exists()
        assert (output_folder / "train_labels.pt").exists()
        assert (output_folder / "val_images.pt").exists()
        assert (output_folder / "val_labels.pt").exists()
        assert (output_folder / "test_images.pt").exists()
        assert (output_folder / "test_labels.pt").exists()
        assert (output_folder / "class_names.pt").exists()

        # Load and validate the data
        train_images = torch.load(output_folder / "train_images.pt")
        class_names = torch.load(output_folder / "class_names.pt")

        assert train_images.shape[0] > 0  # Ensure images are saved
        assert len(class_names) == len(class_labels)  # Ensure class names are correct


# Test the normalize function
def test_normalize():
    images = torch.rand((10, 3, 224, 224))
    normalized_images = normalize(images)
    assert torch.isclose(normalized_images.mean(), torch.tensor(0.0), atol=1e-5)
    assert torch.isclose(normalized_images.std(), torch.tensor(1.0), atol=1e-5)


# Missing test for download_extract_dataset
