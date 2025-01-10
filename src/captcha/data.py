from pathlib import Path
import random
import typer
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import kagglehub


class MyDataset(Dataset):
    """My custom dataset."""

    

    def __init__(self, raw_data_path: Path) -> None:
        """
        Initialize the dataset with the given path to raw data.
        """
        self.data_path = raw_data_path

    def __len__(self) -> int:
        """
        Return the length of the dataset.
        (Implement logic for counting images or data samples, if needed.)
        """
        return len(list(self.data_path.glob("**/*.png")))

    def __getitem__(self, index: int):
        """
        Return the sample (image, label, etc.) at the given index.
        (Implementation depends on your downstream use-case.)
        """
        img_files = list(self.data_path.glob("**/*.png"))
        img_path = img_files[index]
        with Image.open(img_path) as img:
            return img
        
        
  
    def preprocess(self, output_folder: Path) -> None:
        output_folder.mkdir(parents=True, exist_ok=True)

        img_files = list(self.data_path.glob("**/*.png"))
        random.shuffle(img_files)

        # For selecting a subset 
        subset_size = 100
        img_files = img_files[:min(subset_size, len(img_files))]

        # Extracting labels 
        all_labels = []
        for img_path in img_files:
            label_str = img_path.stem.split('_')[0]
            all_labels.append(label_str)
        
        # Convert to a sorted list of unique labels
        class_names = sorted(list(set(all_labels)))
        # Create a dictionary {label_str: class_idx}
        label_to_idx = {lbl: i for i, lbl in enumerate(class_names)}
        
        # Split into train/val/test
        total_count = len(img_files)
        test_count = int(0.10 * total_count)
        val_count = int(0.20 * total_count)
        train_count = total_count - test_count - val_count

        train_files = img_files[:train_count]
        val_files = img_files[train_count : train_count + val_count]
        test_files = img_files[train_count + val_count :]


        # To tensor and Resize to 244x244
        #TODO Normalize 
        transform = transforms.Compose([
            transforms.Resize((52, 32)),
            transforms.ToTensor(),
        ])

        # Return a tuple (image_tensor, label_int)
        def process_split(image_paths):
            images = []
            labels = []
            for img_path in image_paths:
                label_str = img_path.stem.split('_')[0]
                label_int = label_to_idx[label_str]
            with Image.open(img_path) as img:
                img_tensor = transform(img)
            images.append(img_tensor)
            labels.append(label_int)
            return images, labels

        # Split into datasets tensors 
        print(f"Processing {train_count} images for train split...")
        train_images, train_labels = process_split(train_files)
        torch.save(train_images, output_folder / "train_images.pt")
        torch.save(train_labels, output_folder / "train_labels.pt")

        print(f"Processing {val_count} images for validation split...")
        val_images, val_labels = process_split(val_files)
        torch.save(val_images, output_folder / "val_images.pt")
        torch.save(val_labels, output_folder / "val_labels.pt")

        print(f"Processing {test_count} images for test split...")
        test_images, test_labels = process_split(test_files)
        torch.save(test_images, output_folder / "test_images.pt")
        torch.save(test_labels, output_folder / "test_labels.pt")

        # Save class names
        torch.save(class_names, output_folder / "class_names.pt")


        # Summary 
        print("Preprocessing complete.")
        print(f"Processed data saved to: {output_folder}")
        print("Splits:")
        print(f"  Train: {train_count} samples -> train.pt")
        print(f"  Val:   {val_count} samples  -> val.pt")
        print(f"  Test:  {test_count} samples -> test.pt")
        print("Saved class names to: class_names.pt")
        print(f"Found {len(class_names)} unique classes:")

        


def preprocess() -> None:
#TODO FIX DOWNLOAD 


#    print("Downloading dataset from Kaggle...")
    kaggle_path_str = kagglehub.dataset_download("tahabakhtari/captcha-characters-dataset-118k-images")
    kaggle_path = Path(kaggle_path_str)
#
#    print("Path to downloaded dataset files:", kaggle_path)
#    print("Preprocessing data...")
#
#    raw_data_folder = Path("data/raw")
#    raw_data_folder.mkdir(parents=True, exist_ok=True)
#    for file in kaggle_path.glob("**/*.png"):
#        new_path = raw_data_folder / file.name
#        file.replace(new_path)
#
#
#    print("Path to downloaded dataset files:", kaggle_path)
    print("Preprocessing data...")
    raw_data_path = Path("data/raw")
    dataset = MyDataset(raw_data_path)
    dataset.preprocess(Path("data/processed"))


if __name__ == "__main__":
    typer.run(preprocess)

