"""Generate emotion predictions for cat breed dataset and create combined JSON datasets."""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch
import torchvision.transforms as transforms
import yaml
from PIL import Image
from tqdm import tqdm

from model import load_model

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_pipeline_config(config_path: str = "src/configs/pipeline.yaml") -> Dict:
    """Load pipeline configuration from YAML file."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info(f"Pipeline configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise


class EmotionDatasetGenerator:
    """Generate emotion predictions for cat breed dataset."""

    def __init__(self, config: Dict):
        """
        Initialize the emotion dataset generator from configuration.

        Args:
            config: Pipeline configuration dictionary loaded from YAML
        """
        self.config = config
        pipeline_config = config["pipeline"]
        emotion_model_config = config["model_config"]["emotion"]

        self.device = pipeline_config["device"] if torch.cuda.is_available() else "cpu"
        self.img_size = pipeline_config["img_size"]

        # Load class names
        with open(config["class_names"]["emotion_classes"], "r") as f:
            self.emotion_classes = json.load(f)

        with open(config["class_names"]["breed_classes"], "r") as f:
            self.breed_classes = json.load(f)

        self.breed_data_root = Path(config["data"]["breed_data_root"])

        # Load emotion model
        emotion_model_path = config["models"]["emotion_model"]
        logger.info(f"Loading emotion model from {emotion_model_path}")
        logger.info(f"Emotion model config: {emotion_model_config}")
        self.emotion_model = load_model(
            emotion_model_path,
            num_classes=len(self.emotion_classes),
            model_type="emotion",
            model_config=emotion_model_config,
        )
        self.emotion_model.to(self.device)
        self.emotion_model.eval()

        # Set up transforms
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        logger.info(f"EmotionDatasetGenerator initialized using device: {self.device}")

    def predict_emotion(
        self, image_path: Union[str, Path]
    ) -> Tuple[str, float, List[Dict[str, float]]]:
        """
        Predict emotion for a single image.

        Args:
            image_path: Path to the image

        Returns:
            Tuple of (predicted_emotion, confidence, all_predictions)
        """
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert("RGB")
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)

            # Get predictions
            with torch.no_grad():
                outputs = self.emotion_model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]

            # Get all predictions
            probs = probabilities.cpu().numpy()
            all_predictions = [
                {"emotion": self.emotion_classes[i], "score": float(probs[i])}
                for i in range(len(self.emotion_classes))
            ]

            # Get top prediction
            top_idx = torch.argmax(probabilities).item()
            predicted_emotion = self.emotion_classes[top_idx]
            confidence = float(probabilities[top_idx])

            return predicted_emotion, confidence, all_predictions

        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            # Return default prediction in case of error
            return "Other", 0.0, [{"emotion": cls, "score": 0.0} for cls in self.emotion_classes]

    def process_all_images(self) -> List[Dict]:
        """
        Process all images in the breed dataset and generate emotion predictions.

        Returns:
            List of sample dictionaries with breed and emotion information
        """
        samples = []
        total_processed = 0
        total_errors = 0

        logger.info("Starting emotion prediction for all breed images...")

        # Process each breed folder
        for breed_idx, breed_name in enumerate(self.breed_classes):
            breed_folder = self.breed_data_root / breed_name

            if not breed_folder.exists():
                logger.warning(f"Breed folder not found: {breed_folder}")
                continue

            # Get all image files in the breed folder
            image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
            image_files = [
                f for f in breed_folder.iterdir() if f.suffix.lower() in image_extensions
            ]

            logger.info(f"Processing {len(image_files)} images for breed: {breed_name}")

            # Process each image with progress bar
            for image_file in tqdm(image_files, desc=f"Processing {breed_name}"):
                try:
                    # Get emotion prediction
                    predicted_emotion, emotion_confidence, all_emotion_scores = (
                        self.predict_emotion(image_file)
                    )

                    # Get emotion label index
                    emotion_label = self.emotion_classes.index(predicted_emotion)

                    # Create sample entry
                    sample = {
                        "image_path": f"{breed_name}/{image_file.name}",
                        "breed_label": breed_idx,
                        "breed_class": breed_name,
                        "emotion_label": emotion_label,
                        "emotion_class": predicted_emotion,
                        "emotion_confidence": emotion_confidence,
                        "emotion_scores": all_emotion_scores,
                    }

                    samples.append(sample)
                    total_processed += 1

                except Exception as e:
                    logger.error(f"Error processing {image_file}: {e}")
                    total_errors += 1
                    continue

        logger.info(
            f"Processing complete! Total processed: {total_processed}, Errors: {total_errors}"
        )
        return samples

    def save_emotion_predictions(self, samples: List[Dict], output_path: Union[str, Path]) -> None:
        """
        Save emotion predictions to JSON file.

        Args:
            samples: List of sample dictionaries
            output_path: Path to save the JSON file
        """
        output_data = {
            "metadata": {
                "total_samples": len(samples),
                "breed_classes": self.breed_classes,
                "emotion_classes": self.emotion_classes,
                "num_breed_classes": len(self.breed_classes),
                "num_emotion_classes": len(self.emotion_classes),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "samples": samples,
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Emotion predictions saved to {output_path}")

    def create_dataset_splits(
        self,
        samples: List[Dict],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        output_dir: Union[str, Path] = None,
    ) -> Dict[str, List[Dict]]:
        """
        Create train/validation/test splits and save to JSON files.

        Args:
            samples: List of all samples
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            output_dir: Directory to save split files

        Returns:
            Dictionary with train/val/test splits
        """
        # Verify ratios sum to 1
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")

        # Shuffle samples for random split
        import random

        random.seed(42)  # For reproducibility
        shuffled_samples = samples.copy()
        random.shuffle(shuffled_samples)

        # Calculate split indices
        total_samples = len(shuffled_samples)
        train_end = int(total_samples * train_ratio)
        val_end = train_end + int(total_samples * val_ratio)

        # Create splits
        splits = {
            "train": shuffled_samples[:train_end],
            "val": shuffled_samples[train_end:val_end],
            "test": shuffled_samples[val_end:],
        }

        logger.info(f"Dataset splits created:")
        logger.info(f"  Train: {len(splits['train'])} samples")
        logger.info(f"  Validation: {len(splits['val'])} samples")
        logger.info(f"  Test: {len(splits['test'])} samples")

        # Save splits if output directory provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            for split_name, split_samples in splits.items():
                # Create dataset format for each split
                dataset = {
                    "metadata": {
                        "split": split_name,
                        "total_samples": len(split_samples),
                        "breed_classes": self.breed_classes,
                        "emotion_classes": self.emotion_classes,
                        "num_breed_classes": len(self.breed_classes),
                        "num_emotion_classes": len(self.emotion_classes),
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                    "samples": split_samples,
                }

                # Save split file
                split_file = output_dir / f"{split_name}_dataset_with_emotions.json"
                with open(split_file, "w", encoding="utf-8") as f:
                    json.dump(dataset, f, indent=2, ensure_ascii=False)

                logger.info(f"Saved {split_name} split to {split_file}")

        return splits

    def generate_statistics(self, samples: List[Dict]) -> Dict:
        """
        Generate statistics about the dataset.

        Args:
            samples: List of sample dictionaries

        Returns:
            Dictionary with dataset statistics
        """
        # Count samples per breed
        breed_counts = {}
        for sample in samples:
            breed = sample["breed_class"]
            breed_counts[breed] = breed_counts.get(breed, 0) + 1

        # Count samples per emotion
        emotion_counts = {}
        for sample in samples:
            emotion = sample["emotion_class"]
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        # Calculate breed-emotion combinations
        breed_emotion_matrix = {}
        for sample in samples:
            breed = sample["breed_class"]
            emotion = sample["emotion_class"]

            if breed not in breed_emotion_matrix:
                breed_emotion_matrix[breed] = {}
            breed_emotion_matrix[breed][emotion] = breed_emotion_matrix[breed].get(emotion, 0) + 1

        # Calculate average emotion confidence
        total_confidence = sum(sample["emotion_confidence"] for sample in samples)
        avg_confidence = total_confidence / len(samples) if samples else 0

        statistics = {
            "total_samples": len(samples),
            "breed_distribution": breed_counts,
            "emotion_distribution": emotion_counts,
            "breed_emotion_matrix": breed_emotion_matrix,
            "average_emotion_confidence": avg_confidence,
            "breeds_with_most_samples": sorted(
                breed_counts.items(), key=lambda x: x[1], reverse=True
            )[:5],
            "most_common_emotions": sorted(
                emotion_counts.items(), key=lambda x: x[1], reverse=True
            ),
        }

        return statistics


def main():
    """Main function to generate emotion dataset."""
    # Load configuration
    config = load_pipeline_config()

    logger.info(f"Emotion classes: {config['class_names']['emotion_classes']}")
    logger.info(f"Breed classes: {config['class_names']['breed_classes']}")

    # Initialize generator
    generator = EmotionDatasetGenerator(config)

    # Process all images
    logger.info("Starting emotion prediction generation...")
    start_time = time.time()

    samples = generator.process_all_images()

    processing_time = time.time() - start_time
    logger.info(f"Processing completed in {processing_time:.2f} seconds")

    if not samples:
        logger.error("No samples were processed successfully!")
        return

    # Get output directory from config
    output_dir = Path(config["data"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save complete emotion predictions
    generator.save_emotion_predictions(samples, output_dir / "all_samples_with_emotions.json")

    # Create dataset splits
    splits = generator.create_dataset_splits(
        samples, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, output_dir=output_dir
    )

    # Generate and save statistics
    stats = generator.generate_statistics(samples)

    stats_file = output_dir / "dataset_statistics.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info(f"Dataset statistics saved to {stats_file}")

    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("DATASET GENERATION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total samples processed: {len(samples)}")
    logger.info(f"Total breeds: {len(generator.breed_classes)}")
    logger.info(f"Total emotions: {len(generator.emotion_classes)}")
    logger.info(f"Average emotion confidence: {stats['average_emotion_confidence']:.3f}")
    logger.info(f"Processing time: {processing_time:.2f} seconds")

    logger.info("\nEmotion distribution:")
    for emotion, count in stats["most_common_emotions"]:
        percentage = (count / len(samples)) * 100
        logger.info(f"  {emotion}: {count} ({percentage:.1f}%)")

    logger.info("\nTop 5 breeds by sample count:")
    for breed, count in stats["breeds_with_most_samples"]:
        logger.info(f"  {breed}: {count} samples")

    logger.info(f"\nFiles saved to: {output_dir}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
