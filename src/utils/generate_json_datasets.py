#!/usr/bin/env python3
"""
Script to generate JSON dataset files from directory-based datasets.
Supports single-task (breed-only, emotion-only) and multitask (breed+emotion) generation.
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
from sklearn.model_selection import train_test_split


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


logger = logging.getLogger(__name__)


def scan_directory(data_dir: Path) -> Dict[str, List[str]]:
    """
    Scan directory for images organized by class folders.

    Args:
        data_dir: Path to data directory containing class folders

    Returns:
        Dictionary mapping class names to lists of image paths
    """
    class_to_images = {}

    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")

    # Get all subdirectories (class folders)
    class_dirs = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]

    if not class_dirs:
        raise ValueError(f"No class directories found in {data_dir}")

    # Common image extensions
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}

    for class_dir in class_dirs:
        class_name = class_dir.name
        image_files = []

        # Find all image files in the class directory
        for img_file in class_dir.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in image_extensions:
                # Store relative path from data_dir
                relative_path = img_file.relative_to(data_dir)
                image_files.append(str(relative_path))

        if image_files:
            class_to_images[class_name] = image_files
            logger.info(f"Found {len(image_files)} images for class '{class_name}'")

    return class_to_images


def create_single_task_samples(
    class_to_images: Dict[str, List[str]], task_type: str
) -> Tuple[List[Dict], List[str]]:
    """
    Create samples for single-task dataset.

    Args:
        class_to_images: Dictionary mapping class names to image paths
        task_type: Either "breed" or "emotion"

    Returns:
        Tuple of (samples list, class names list)
    """
    samples = []
    class_names = sorted(class_to_images.keys())

    for class_idx, class_name in enumerate(class_names):
        for image_path in class_to_images[class_name]:
            sample = {"image_path": image_path, "label": class_idx, "class_name": class_name}
            samples.append(sample)

    logger.info(
        f"Created {len(samples)} samples for {task_type} task with {len(class_names)} classes"
    )
    return samples, class_names


def assign_random_emotions(
    breed_samples: List[Dict],
    emotion_classes: List[str],
    emotion_distribution: Dict[str, float] = None,
) -> List[Dict]:
    """
    Assign random emotions to breed samples for multitask learning.

    Args:
        breed_samples: List of breed samples
        emotion_classes: List of emotion class names
        emotion_distribution: Optional distribution weights for emotions

    Returns:
        List of samples with both breed and emotion labels
    """
    if emotion_distribution is None:
        # Default uniform distribution
        emotion_distribution = {emotion: 1.0 for emotion in emotion_classes}

    # Normalize distribution
    total_weight = sum(emotion_distribution.values())
    normalized_weights = [
        emotion_distribution.get(emotion, 1.0) / total_weight for emotion in emotion_classes
    ]

    multitask_samples = []

    for sample in breed_samples:
        # Randomly assign emotion based on distribution
        emotion_idx = np.random.choice(len(emotion_classes), p=normalized_weights)
        emotion_name = emotion_classes[emotion_idx]

        multitask_sample = {
            "image_path": sample["image_path"],
            "breed_label": sample["label"],
            "breed_class": sample["class_name"],
            "emotion_label": emotion_idx,
            "emotion_class": emotion_name,
        }
        multitask_samples.append(multitask_sample)

    logger.info(f"Created {len(multitask_samples)} multitask samples")
    return multitask_samples


def create_multitask_samples_from_directories(
    breed_data_dir: Path, emotion_data_dir: Path, strategy: str = "random_assignment"
) -> Tuple[List[Dict], List[str], List[str]]:
    """
    Create multitask samples from separate breed and emotion directories.

    Args:
        breed_data_dir: Path to breed data directory
        emotion_data_dir: Path to emotion data directory
        strategy: Strategy for combining breed and emotion data

    Returns:
        Tuple of (samples, breed_classes, emotion_classes)
    """
    # Scan both directories
    breed_class_to_images = scan_directory(breed_data_dir)
    emotion_class_to_images = scan_directory(emotion_data_dir)

    # Create breed samples
    breed_samples, breed_classes = create_single_task_samples(breed_class_to_images, "breed")
    emotion_samples, emotion_classes = create_single_task_samples(
        emotion_class_to_images, "emotion"
    )

    if strategy == "random_assignment":
        # Assign random emotions to breed images
        multitask_samples = assign_random_emotions(breed_samples, emotion_classes)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return multitask_samples, breed_classes, emotion_classes


def split_dataset_samples(
    samples: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split samples into train, validation, and test sets.

    Args:
        samples: List of samples to split
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        random_state: Random state for reproducibility

    Returns:
        Tuple of (train_samples, val_samples, test_samples)
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")

    # Shuffle samples
    random.seed(random_state)
    shuffled_samples = samples.copy()
    random.shuffle(shuffled_samples)

    # First split: separate test set
    train_val_samples, test_samples = train_test_split(
        shuffled_samples, test_size=test_ratio, random_state=random_state
    )

    # Second split: separate train and validation
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train_samples, val_samples = train_test_split(
        train_val_samples, test_size=val_ratio_adjusted, random_state=random_state
    )

    logger.info(
        f"Split: {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test"
    )
    return train_samples, val_samples, test_samples


def create_dataset_json(samples: List[Dict], split: str, metadata: Dict = None) -> Dict:
    """
    Create dataset JSON structure.

    Args:
        samples: List of samples
        split: Dataset split name (train/val/test)
        metadata: Optional metadata to include

    Returns:
        Dataset dictionary
    """
    dataset = {"split": split, "total_samples": len(samples), "samples": samples}

    if metadata:
        dataset["metadata"] = metadata

    return dataset


def generate_single_task_datasets(
    data_dir: Path,
    output_dir: Path,
    task_type: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> None:
    """
    Generate single-task JSON datasets from directory structure.

    Args:
        data_dir: Path to data directory
        output_dir: Path to output directory
        task_type: Task type ("breed" or "emotion")
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_state: Random state for reproducibility
    """
    logger.info(f"Generating {task_type} datasets from {data_dir}")

    # Scan directory
    class_to_images = scan_directory(data_dir)

    # Create samples
    samples, class_names = create_single_task_samples(class_to_images, task_type)

    # Split samples
    train_samples, val_samples, test_samples = split_dataset_samples(
        samples, train_ratio, val_ratio, test_ratio, random_state
    )

    # Create metadata
    metadata = {
        "task_type": task_type,
        "class_names": class_names,
        "num_classes": len(class_names),
        "data_source": str(data_dir),
    }

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save datasets
    splits = [("train", train_samples), ("val", val_samples), ("test", test_samples)]

    for split_name, split_samples in splits:
        dataset = create_dataset_json(split_samples, split_name, metadata)

        output_file = output_dir / f"{split_name}_dataset.json"
        with open(output_file, "w") as f:
            json.dump(dataset, f, indent=2)

        logger.info(f"Saved {split_name} dataset: {output_file} ({len(split_samples)} samples)")


def generate_multitask_datasets(
    breed_data_dir: Path,
    emotion_data_dir: Path,
    output_dir: Path,
    strategy: str = "random_assignment",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> None:
    """
    Generate multitask JSON datasets from breed and emotion directories.

    Args:
        breed_data_dir: Path to breed data directory
        emotion_data_dir: Path to emotion data directory
        output_dir: Path to output directory
        strategy: Strategy for combining data
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_state: Random state for reproducibility
    """
    logger.info(
        f"Generating multitask datasets from breed: {breed_data_dir}, emotion: {emotion_data_dir}"
    )

    # Create multitask samples
    samples, breed_classes, emotion_classes = create_multitask_samples_from_directories(
        breed_data_dir, emotion_data_dir, strategy
    )

    # Split samples
    train_samples, val_samples, test_samples = split_dataset_samples(
        samples, train_ratio, val_ratio, test_ratio, random_state
    )

    # Create metadata
    metadata = {
        "task_type": "multitask",
        "breed_classes": breed_classes,
        "emotion_classes": emotion_classes,
        "num_breed_classes": len(breed_classes),
        "num_emotion_classes": len(emotion_classes),
        "breed_data_source": str(breed_data_dir),
        "emotion_data_source": str(emotion_data_dir),
        "combination_strategy": strategy,
    }

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save datasets
    splits = [("train", train_samples), ("val", val_samples), ("test", test_samples)]

    for split_name, split_samples in splits:
        dataset = create_dataset_json(split_samples, split_name, metadata)

        output_file = output_dir / f"{split_name}_dataset.json"
        with open(output_file, "w") as f:
            json.dump(dataset, f, indent=2)

        logger.info(f"Saved {split_name} dataset: {output_file} ({len(split_samples)} samples)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate JSON dataset files from directory-based datasets"
    )

    # Mode selection
    parser.add_argument(
        "mode", choices=["breed", "emotion", "multitask"], help="Type of dataset to generate"
    )

    # Data directories
    parser.add_argument(
        "--breed-data-dir",
        type=Path,
        help="Path to breed data directory (required for breed and multitask modes)",
    )
    parser.add_argument(
        "--emotion-data-dir",
        type=Path,
        help="Path to emotion data directory (required for emotion and multitask modes)",
    )

    # Output
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Output directory for JSON files"
    )

    # Split ratios
    parser.add_argument(
        "--train-ratio", type=float, default=0.7, help="Training set ratio (default: 0.7)"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.15, help="Validation set ratio (default: 0.15)"
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.15, help="Test set ratio (default: 0.15)"
    )

    # Other options
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="random_assignment",
        choices=["random_assignment"],
        help="Strategy for multitask data combination (default: random_assignment)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.log_level)

    # Validate arguments
    if args.mode in ["breed", "multitask"] and not args.breed_data_dir:
        parser.error(f"--breed-data-dir is required for {args.mode} mode")

    if args.mode in ["emotion", "multitask"] and not args.emotion_data_dir:
        parser.error(f"--emotion-data-dir is required for {args.mode} mode")

    # Validate split ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        parser.error("Train, validation, and test ratios must sum to 1.0")

    # Generate datasets
    try:
        if args.mode == "breed":
            generate_single_task_datasets(
                data_dir=args.breed_data_dir,
                output_dir=args.output_dir,
                task_type="breed",
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                random_state=args.random_state,
            )
        elif args.mode == "emotion":
            generate_single_task_datasets(
                data_dir=args.emotion_data_dir,
                output_dir=args.output_dir,
                task_type="emotion",
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                random_state=args.random_state,
            )
        elif args.mode == "multitask":
            generate_multitask_datasets(
                breed_data_dir=args.breed_data_dir,
                emotion_data_dir=args.emotion_data_dir,
                output_dir=args.output_dir,
                strategy=args.strategy,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                random_state=args.random_state,
            )

        logger.info(f"Successfully generated {args.mode} datasets in {args.output_dir}")

    except Exception as e:
        logger.error(f"Error generating datasets: {e}")
        raise


if __name__ == "__main__":
    main()
