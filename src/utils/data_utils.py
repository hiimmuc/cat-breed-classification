"""Data loaders and utilities for the cat breed classification project."""

import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


def print_dataset_stats(data_loaders: Dict[str, Any]) -> None:
    """Print concise dataset statistics."""
    train_loader, val_loader, test_loader = (
        data_loaders["train"],
        data_loaders["val"],
        data_loaders["test"],
    )
    class_names = data_loaders["class_names"]

    # Quick class distribution count
    class_counts = {}
    for _, labels in train_loader:
        for label in labels:
            class_name = class_names[label.item()]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

    # Print summary
    total_samples = len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)
    print(f"\n� Dataset Summary �")
    print(f"{'='*50}")
    print(f"Classes: {len(class_names)} | Total samples: {total_samples}")
    print(
        f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}"
    )
    print(f"Batch size: {train_loader.batch_size}")

    # Show top classes
    if class_counts:
        top_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\nTop 5 classes: {', '.join([f'{name}({count})' for name, count in top_classes])}")
    print(f"{'='*50}\n")


def print_training_config(config: Dict[str, Any]) -> None:
    """Print training configuration in a concise format."""
    print(f"\n⚙️  Training Configuration ⚙️")
    print(f"{'='*50}")

    # Group and format key config items
    important_keys = ["backbone", "epochs", "batch_size", "learning_rate", "device"]
    config_items = [f"{key}: {config.get(key, 'N/A')}" for key in important_keys if key in config]

    for item in config_items:
        print(f"  {item}")

    print(f"{'='*50}\n")


class CatBreedDataset(Dataset):
    """Dataset class for loading cat breed images from JSON dataset files."""

    def __init__(
        self,
        json_path: Optional[Union[str, Path]] = None,
        data_dir: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
        split: str = "train",
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42,
    ):
        """
        Initialize the CatBreedDataset.

        Args:
            json_path: Path to JSON dataset file (preferred method)
            data_dir: Directory containing class folders with images (legacy method)
            transform: Transforms to apply to the images
            split: One of 'train', 'val', or 'test' (only used with data_dir)
            val_ratio: Ratio of data to use for validation (only used with data_dir)
            test_ratio: Ratio of data to use for testing (only used with data_dir)
            random_seed: Random seed for reproducibility (only used with data_dir)
        """
        self.transform = transform

        if json_path is not None:
            # Load from JSON file (preferred method)
            self._load_from_json(json_path)
        elif data_dir is not None:
            # Legacy method - load directly from directory
            logger.warning(
                "Loading from directory is deprecated. Consider using JSON dataset files."
            )
            self._load_from_directory(data_dir, split, val_ratio, test_ratio, random_seed)
        else:
            raise ValueError("Either json_path or data_dir must be provided")

    def _load_from_json(self, json_path: Union[str, Path]) -> None:
        """Load dataset from JSON file."""
        dataset = load_dataset_json(json_path)

        self.data_dir = Path(dataset["metadata"]["data_dir"])
        self.class_names = dataset["metadata"]["class_names"]
        self.class_to_idx = dataset["metadata"]["class_to_idx"]
        self.split = dataset["split"]

        # Extract image paths and labels
        self.samples = dataset["samples"]
        logger.info(f"Loaded {self.split} dataset with {len(self.samples)} samples from JSON")

    def _load_from_directory(
        self,
        data_dir: Union[str, Path],
        split: str,
        val_ratio: float,
        test_ratio: float,
        random_seed: int,
    ) -> None:
        """Legacy method to load dataset directly from directory."""
        self.data_dir = Path(data_dir)
        self.split = split

        self.class_names = [
            d.name for d in self.data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
        ]
        self.class_names.sort()  # Ensure consistent ordering
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}

        images, labels = self._load_dataset_legacy()

        # Split dataset using random seed for reproducibility
        np.random.seed(random_seed)
        indices = np.random.permutation(len(images))

        test_size = int(len(indices) * test_ratio)
        val_size = int(len(indices) * val_ratio)
        train_size = len(indices) - test_size - val_size

        if split == "train":
            selected_indices = indices[:train_size]
        elif split == "val":
            selected_indices = indices[train_size : train_size + val_size]
        else:  # test
            selected_indices = indices[train_size + val_size :]

        # Create samples list
        self.samples = []
        for idx in selected_indices:
            img_path = images[idx]
            relative_path = str(img_path.relative_to(self.data_dir))
            self.samples.append(
                {
                    "image_path": relative_path,
                    "label": labels[idx],
                    "class_name": self.class_names[labels[idx]],
                }
            )

        logger.info(f"Created {split} dataset with {len(self.samples)} samples")

    def _load_dataset_legacy(self) -> Tuple[List[Path], List[int]]:
        """Load dataset paths and labels (legacy method)."""
        images = []
        labels = []

        for class_name in tqdm(self.class_names, desc="Loading dataset"):
            class_dir = self.data_dir / class_name
            class_idx = self.class_to_idx[class_name]

            # Support multiple image formats
            image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
            for pattern in image_extensions:
                for img_path in class_dir.glob(pattern):
                    images.append(img_path)
                    labels.append(class_idx)

        return images, labels

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Tuple containing the image tensor and class label
        """
        sample = self.samples[idx]
        img_path = self.data_dir / sample["image_path"]
        label = sample["label"]

        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            img = Image.new("RGB", (224, 224), color="black")

        if self.transform:
            img = self.transform(img)

        return img, label


def get_data_loaders(
    data_dir: Optional[Union[str, Path]] = None,
    json_dir: Optional[Union[str, Path]] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 224,
    augment_data: bool = True,
) -> Dict[str, DataLoader]:
    """
    Create data loaders for training, validation and testing.

    Args:
        data_dir: Directory containing class folders with images (legacy method)
        json_dir: Directory containing JSON dataset files (preferred method)
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
        img_size: Size of the input images

    Returns:
        Dictionary containing train, val, and test data loaders
    """
    if json_dir is not None and data_dir is not None:
        raise ValueError("Provide either json_dir or data_dir, not both")

    if json_dir is None and data_dir is None:
        raise ValueError("Either json_dir or data_dir must be provided")

    # Define transforms
    if augment_data:
        train_transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    val_test_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create datasets
    if json_dir is not None:
        # Load from JSON files (preferred method)
        json_dir = Path(json_dir)
        train_dataset = CatBreedDataset(
            json_path=json_dir / "train_dataset.json", transform=train_transform
        )
        val_dataset = CatBreedDataset(
            json_path=json_dir / "val_dataset.json", transform=val_test_transform
        )
        test_dataset = CatBreedDataset(
            json_path=json_dir / "test_dataset.json", transform=val_test_transform
        )
    else:
        # Legacy method - load from directory
        train_dataset = CatBreedDataset(
            data_dir=data_dir, transform=train_transform, split="train"
        )
        val_dataset = CatBreedDataset(data_dir=data_dir, transform=val_test_transform, split="val")
        test_dataset = CatBreedDataset(
            data_dir=data_dir, transform=val_test_transform, split="test"
        )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    data_loaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "class_names": train_dataset.class_names,
        "class_to_idx": train_dataset.class_to_idx,
    }

    # Print dataset statistics
    print_dataset_stats(data_loaders)

    return data_loaders


def get_test_loader_from_json(
    json_path: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 224,
) -> Dict[str, Any]:
    """
    Create a test data loader from a JSON dataset file.

    Args:
        json_path: Path to the JSON dataset file
        batch_size: Batch size for data loader
        num_workers: Number of workers for data loading
        img_size: Size of the input images

    Returns:
        Dictionary containing test loader and metadata
    """
    # Define transform for testing
    test_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create dataset
    test_dataset = CatBreedDataset(json_path=json_path, transform=test_transform)

    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return {
        "test_loader": test_loader,
        "dataset": test_dataset,
        "class_names": test_dataset.class_names,
        "class_to_idx": test_dataset.class_to_idx,
        "split": test_dataset.split,
        "num_samples": len(test_dataset),
    }


def visualize_batch(
    batch: Tuple[torch.Tensor, torch.Tensor],
    class_names: List[str],
    n_samples: int = 16,
    title: str = "Sample Batch",
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Visualize a batch of images.

    Args:
        batch: Tuple of (images, labels)
        class_names: List of class names
        n_samples: Number of samples to display
        title: Title for the plot
        save_path: Optional path to save the visualization
    """
    images, labels = batch
    images = images[:n_samples]
    labels = labels[:n_samples]

    # Denormalize images
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    images = images * std + mean
    images = torch.clamp(images, 0, 1)

    # Plot images in a grid
    n_cols = 4
    n_rows = (n_samples + n_cols - 1) // n_cols

    plt.figure(figsize=(15, 3 * n_rows))
    for i in range(n_samples):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(images[i].permute(1, 2, 0).cpu().numpy())
        plt.title(class_names[labels[i]])
        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Visualization saved to: {save_path}")

    plt.show()


def analyze_dataset_json(json_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Analyze a dataset JSON file and return statistics.

    Args:
        json_path: Path to the JSON dataset file

    Returns:
        Dictionary containing dataset analysis
    """
    dataset = load_dataset_json(json_path)

    # Count samples per class
    class_counts = {}
    for sample in dataset["samples"]:
        class_name = sample["class_name"]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    # Calculate statistics
    total_samples = len(dataset["samples"])
    num_classes = len(class_counts)
    avg_samples_per_class = total_samples / num_classes if num_classes > 0 else 0
    min_samples = min(class_counts.values()) if class_counts else 0
    max_samples = max(class_counts.values()) if class_counts else 0

    analysis = {
        "split": dataset["split"],
        "total_samples": total_samples,
        "num_classes": num_classes,
        "class_counts": class_counts,
        "avg_samples_per_class": avg_samples_per_class,
        "min_samples_per_class": min_samples,
        "max_samples_per_class": max_samples,
        "metadata": dataset["metadata"],
    }

    return analysis


def print_dataset_analysis(analysis: Dict[str, Any]) -> None:
    """
    Print dataset analysis in a formatted way.

    Args:
        analysis: Dictionary containing dataset analysis
    """
    print(f"\n📊 Dataset Analysis: {analysis['split'].upper()}")
    print("=" * 50)
    print(f"Total samples: {analysis['total_samples']:,}")
    print(f"Number of classes: {analysis['num_classes']}")
    print(f"Average samples per class: {analysis['avg_samples_per_class']:.1f}")
    print(f"Min samples per class: {analysis['min_samples_per_class']}")
    print(f"Max samples per class: {analysis['max_samples_per_class']}")

    print(f"\n📈 Class Distribution:")
    print("-" * 30)
    sorted_classes = sorted(analysis["class_counts"].items(), key=lambda x: x[1], reverse=True)

    for i, (class_name, count) in enumerate(sorted_classes):
        percentage = (count / analysis["total_samples"]) * 100
        print(f"{class_name:<20} {count:>6} ({percentage:>5.1f}%)")

        # Show only top 10 if many classes
        if i >= 9 and len(sorted_classes) > 12:
            remaining = len(sorted_classes) - 10
            print(f"... and {remaining} more classes")
            break

    metadata = analysis["metadata"]
    print(f"\n📋 Metadata:")
    print(f"Created: {metadata.get('created_at', 'Unknown')}")
    print(f"Random seed: {metadata.get('random_seed', 'Unknown')}")
    print(f"Data directory: {metadata.get('data_dir', 'Unknown')}")
    print("=" * 50)


def create_dataset_json(
    data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Create dataset JSON files for train, validation, test, and total datasets.

    Args:
        data_dir: Directory containing class folders with images
        output_dir: Directory to save the JSON files
        val_ratio: Ratio of data to use for validation
        test_ratio: Ratio of data to use for testing
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary containing dataset statistics
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating dataset JSON files from {data_dir}")

    # Get class names and create mapping
    class_names = [d.name for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    class_names.sort()  # Ensure consistent ordering
    class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}

    # Load all images and labels
    all_samples = []
    for class_name in tqdm(class_names, desc="Scanning images"):
        class_dir = data_dir / class_name
        class_idx = class_to_idx[class_name]

        # Support multiple image formats
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
        for pattern in image_extensions:
            for img_path in class_dir.glob(pattern):
                relative_path = str(img_path.relative_to(data_dir))
                all_samples.append(
                    {"image_path": relative_path, "label": class_idx, "class_name": class_name}
                )

    logger.info(f"Found {len(all_samples)} total samples across {len(class_names)} classes")

    # Split dataset using random seed for reproducibility
    np.random.seed(random_seed)
    indices = np.random.permutation(len(all_samples))

    test_size = int(len(indices) * test_ratio)
    val_size = int(len(indices) * val_ratio)
    train_size = len(indices) - test_size - val_size

    # Create splits
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    # Create dataset dictionaries
    datasets = {
        "train": {
            "samples": [all_samples[i] for i in train_indices],
            "split": "train",
            "size": len(train_indices),
        },
        "val": {
            "samples": [all_samples[i] for i in val_indices],
            "split": "val",
            "size": len(val_indices),
        },
        "test": {
            "samples": [all_samples[i] for i in test_indices],
            "split": "test",
            "size": len(test_indices),
        },
        "total": {"samples": all_samples, "split": "total", "size": len(all_samples)},
    }

    # Add metadata to all datasets
    metadata = {
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "num_classes": len(class_names),
        "data_dir": str(data_dir),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "random_seed": random_seed,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
    }

    # Save JSON files
    for split_name, dataset in datasets.items():
        dataset["metadata"] = metadata
        json_path = output_dir / f"{split_name}_dataset.json"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {split_name} dataset with {dataset['size']} samples to {json_path}")

    # Print statistics
    stats = {
        "total_samples": len(all_samples),
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
        "test_samples": len(test_indices),
        "num_classes": len(class_names),
        "class_names": class_names,
    }

    print(f"\n📊 Dataset JSON Creation Complete 📊")
    print(f"{'=' * 50}")
    print(f"{'Total samples:':<20} {stats['total_samples']:>10}")
    print(f"{'Training samples:':<20} {stats['train_samples']:>10}")
    print(f"{'Validation samples:':<20} {stats['val_samples']:>10}")
    print(f"{'Test samples:':<20} {stats['test_samples']:>10}")
    print(f"{'Number of classes:':<20} {stats['num_classes']:>10}")
    print(f"{'=' * 50}\n")

    return stats


def load_dataset_json(json_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load dataset from JSON file.

    Args:
        json_path: Path to the JSON dataset file

    Returns:
        Dictionary containing dataset information
    """
    json_path = Path(json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    logger.info(
        f"Loaded {dataset['split']} dataset with {dataset['size']} samples from {json_path}"
    )

    return dataset


class CatJSONDataset(Dataset):
    """Dataset that loads from JSON files with breed and emotion labels."""

    def __init__(
        self,
        json_path: Union[str, Path],
        data_root: Union[str, Path],
        transform: Optional[Callable] = None,
        task: str = "multitask",
    ):
        """
        Initialize JSON dataset.

        Args:
            json_path: Path to JSON dataset file
            data_root: Root directory containing images
            transform: Image transformations
            task: Task type ("breed", "emotion", "multitask")
        """
        self.json_path = Path(json_path)
        self.data_root = Path(data_root)
        self.transform = transform
        self.task = task

        # Load JSON data
        with open(self.json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.metadata = self.data["metadata"]
        self.samples = self.data["samples"]

        # Extract class information based on task type
        if self.task == "multitask":
            self.breed_classes = self.metadata["breed_classes"]
            self.emotion_classes = self.metadata["emotion_classes"]
            self.num_breed_classes = len(self.breed_classes)
            self.num_emotion_classes = len(self.emotion_classes)
            logger.info(
                f"Loaded JSON multitask dataset from {self.json_path} with {len(self.samples)} samples"
            )
            logger.info(
                f"Classes: {self.num_breed_classes} breeds, {self.num_emotion_classes} emotions"
            )
        elif self.task == "breed":
            # For breed-only task, use class_names as breed_classes
            self.breed_classes = self.metadata.get("class_names", [])
            self.emotion_classes = []
            self.num_breed_classes = len(self.breed_classes)
            self.num_emotion_classes = 0
            logger.info(
                f"Loaded JSON breed dataset from {self.json_path} with {len(self.samples)} samples"
            )
            logger.info(f"Classes: {self.num_breed_classes} breeds")
        elif self.task == "emotion":
            # For emotion-only task, use class_names as emotion_classes
            self.breed_classes = []
            self.emotion_classes = self.metadata.get("class_names", [])
            self.num_breed_classes = 0
            self.num_emotion_classes = len(self.emotion_classes)
            logger.info(
                f"Loaded JSON emotion dataset from {self.json_path} with {len(self.samples)} samples"
            )
            logger.info(f"Classes: {self.num_emotion_classes} emotions")
        else:
            raise ValueError(f"Unsupported task type: {self.task}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, int, int]]:
        """
        Get sample from dataset.

        Args:
            idx: Sample index

        Returns:
            For single task: (image, label)
            For multitask: (image, breed_label, emotion_label)
        """
        sample = self.samples[idx]

        # Load image
        image_path = self.data_root / sample["image_path"]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}")
            # Create a dummy image if loading fails
            image = Image.new("RGB", (224, 224), (128, 128, 128))

        if self.transform:
            image = self.transform(image)

        # Return based on task type
        if self.task == "breed":
            # For single-task breed datasets, use 'label' field
            label = sample.get("label", sample.get("breed_label", 0))
            return image, label
        elif self.task == "emotion":
            # For single-task emotion datasets, use 'label' field
            label = sample.get("label", sample.get("emotion_label", 0))
            return image, label
        elif self.task == "multitask":
            return image, sample["breed_label"], sample["emotion_label"]
        else:
            raise ValueError(f"Invalid task: {self.task}")


class MultiTaskCatDataset(Dataset):
    """Dataset for multitask cat breed and emotion classification."""

    def __init__(
        self,
        breed_dataset: Dataset,
        emotion_dataset: Dataset,
        transform: Optional[Callable] = None,
        balance_strategy: str = "cycle",
    ):
        """
        Initialize multitask dataset.

        Args:
            breed_dataset: Breed classification dataset
            emotion_dataset: Emotion classification dataset
            transform: Image transformations
            balance_strategy: How to balance datasets ("cycle", "repeat_smaller", "truncate")
        """
        self.breed_dataset = breed_dataset
        self.emotion_dataset = emotion_dataset
        self.transform = transform
        self.balance_strategy = balance_strategy

        # Determine dataset length based on balance strategy
        breed_len = len(breed_dataset)
        emotion_len = len(emotion_dataset)

        if balance_strategy == "cycle":
            # Use the larger dataset length, cycle through smaller one
            self.length = max(breed_len, emotion_len)
        elif balance_strategy == "repeat_smaller":
            # Repeat smaller dataset to match larger one
            self.length = max(breed_len, emotion_len)
        elif balance_strategy == "truncate":
            # Use smaller dataset length
            self.length = min(breed_len, emotion_len)
        else:
            raise ValueError(f"Unknown balance_strategy: {balance_strategy}")

        logger.info(
            f"Created multitask dataset with {self.length} samples "
            f"(breed: {breed_len}, emotion: {emotion_len}, strategy: {balance_strategy})"
        )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        """
        Get sample with both breed and emotion labels.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image, breed_label, emotion_label)
        """
        # Get breed sample
        breed_idx = idx % len(self.breed_dataset)
        breed_image, breed_label = self.breed_dataset[breed_idx]

        # Get emotion sample
        emotion_idx = idx % len(self.emotion_dataset)
        emotion_image, emotion_label = self.emotion_dataset[emotion_idx]

        # For simplicity, use breed image as the primary image
        # In practice, you might want to ensure both datasets have overlapping samples
        image = breed_image

        if self.transform:
            image = self.transform(image)

        return image, breed_label, emotion_label


def get_multitask_data_loaders(
    breed_data_dir: Union[str, Path],
    emotion_data_dir: Union[str, Path],
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    validation_split: float = 0.2,
    test_split: float = 0.1,
    random_seed: int = 42,
    balance_strategy: str = "cycle",
    augment_data: bool = True,
) -> Dict[str, Any]:
    """
    Create data loaders for multitask learning with breed and emotion datasets.

    Args:
        breed_data_dir: Path to breed classification data
        emotion_data_dir: Path to emotion classification data
        img_size: Input image size
        batch_size: Batch size
        num_workers: Number of data loading workers
        validation_split: Fraction for validation set
        test_split: Fraction for test set
        random_seed: Random seed for reproducibility
        balance_strategy: How to balance datasets ("cycle", "repeat_smaller", "truncate")
        augment_data: Whether to apply data augmentation

    Returns:
        Dictionary containing train/val/test loaders and metadata
    """
    logger.info("Creating multitask data loaders...")

    # Get individual dataset loaders
    breed_loaders = get_data_loaders(
        data_dir=breed_data_dir,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        validation_split=validation_split,
        test_split=test_split,
        random_seed=random_seed,
        augment_data=augment_data,
    )

    emotion_loaders = get_data_loaders(
        data_dir=emotion_data_dir,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        validation_split=validation_split,
        test_split=test_split,
        random_seed=random_seed,
        augment_data=augment_data,
    )

    # Create multitask datasets
    multitask_datasets = {}
    for split in ["train", "val", "test"]:
        breed_dataset = breed_loaders[split].dataset
        emotion_dataset = emotion_loaders[split].dataset

        multitask_datasets[split] = MultiTaskCatDataset(
            breed_dataset=breed_dataset,
            emotion_dataset=emotion_dataset,
            balance_strategy=balance_strategy,
        )

    # Create multitask data loaders
    multitask_loaders = {}
    for split, dataset in multitask_datasets.items():
        shuffle = split == "train"
        multitask_loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    # Combine metadata
    result = {
        "train": multitask_loaders["train"],
        "val": multitask_loaders["val"],
        "test": multitask_loaders["test"],
        "breed_class_names": breed_loaders["class_names"],
        "emotion_class_names": emotion_loaders["class_names"],
        "num_breed_classes": len(breed_loaders["class_names"]),
        "num_emotion_classes": len(emotion_loaders["class_names"]),
        "breed_loaders": breed_loaders,  # Keep individual loaders for reference
        "emotion_loaders": emotion_loaders,
    }

    logger.info(
        f"Created multitask loaders with {len(breed_loaders['class_names'])} breed classes "
        f"and {len(emotion_loaders['class_names'])} emotion classes"
    )

    return result


def get_multitask_data_loaders_from_json(
    train_json_path: Union[str, Path],
    val_json_path: Union[str, Path],
    test_json_path: Union[str, Path],
    data_root: Union[str, Path],
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    augment_data: bool = True,
) -> Dict[str, Any]:
    """
    Create multitask data loaders from JSON dataset files.

    Args:
        train_json_path: Path to training JSON dataset
        val_json_path: Path to validation JSON dataset
        test_json_path: Path to test JSON dataset
        data_root: Root directory containing images
        img_size: Input image size
        batch_size: Batch size
        num_workers: Number of data loading workers
        augment_data: Whether to apply data augmentation

    Returns:
        Dictionary containing train/val/test loaders and metadata
    """
    logger.info("Creating multitask data loaders from JSON files...")

    # Define transforms
    if augment_data:
        train_transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    val_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create datasets
    train_dataset = CatJSONDataset(
        json_path=train_json_path, data_root=data_root, transform=train_transform, task="multitask"
    )

    val_dataset = CatJSONDataset(
        json_path=val_json_path, data_root=data_root, transform=val_transform, task="multitask"
    )

    test_dataset = CatJSONDataset(
        json_path=test_json_path, data_root=data_root, transform=val_transform, task="multitask"
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Extract metadata from first dataset (should be consistent across splits)
    result = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "breed_class_names": train_dataset.breed_classes,
        "emotion_class_names": train_dataset.emotion_classes,
        "num_breed_classes": train_dataset.num_breed_classes,
        "num_emotion_classes": train_dataset.num_emotion_classes,
    }

    logger.info(
        f"Created JSON multitask loaders with {train_dataset.num_breed_classes} breed classes "
        f"and {train_dataset.num_emotion_classes} emotion classes"
    )
    logger.info(
        f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    return result


def get_single_task_data_loaders_from_json(
    train_json_path: Union[str, Path],
    val_json_path: Union[str, Path],
    test_json_path: Union[str, Path],
    data_root: Union[str, Path],
    task: str = "breed",
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    augment_data: bool = True,
) -> Dict[str, Any]:
    """
    Create single-task data loaders from JSON dataset files.

    Args:
        train_json_path: Path to training JSON dataset
        val_json_path: Path to validation JSON dataset
        test_json_path: Path to test JSON dataset
        data_root: Root directory containing images
        task: Task type ("breed" or "emotion")
        img_size: Input image size
        batch_size: Batch size
        num_workers: Number of data loading workers
        augment_data: Whether to apply data augmentation

    Returns:
        Dictionary containing train/val/test loaders and metadata
    """
    logger.info(f"Creating {task} data loaders from JSON files...")

    # Define transforms
    if augment_data:
        train_transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    val_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create datasets
    train_dataset = CatJSONDataset(
        json_path=train_json_path, data_root=data_root, transform=train_transform, task=task
    )

    val_dataset = CatJSONDataset(
        json_path=val_json_path, data_root=data_root, transform=val_transform, task=task
    )

    test_dataset = CatJSONDataset(
        json_path=test_json_path, data_root=data_root, transform=val_transform, task=task
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Extract metadata
    if task == "breed":
        class_names = train_dataset.breed_classes
        num_classes = train_dataset.num_breed_classes
    elif task == "emotion":
        class_names = train_dataset.emotion_classes
        num_classes = train_dataset.num_emotion_classes
    else:
        raise ValueError(f"Invalid task: {task}")

    result = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "class_names": class_names,
        "num_classes": num_classes,
    }

    logger.info(f"Created JSON {task} loaders with {num_classes} classes")
    logger.info(
        f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    return result
