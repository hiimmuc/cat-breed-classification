"""Main entry point for cat breed classification."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.optim as optim
import yaml

HOME = Path(__file__).resolve().parent

# Import local modules
sys.path.insert(0, str(HOME))
from engine.evaluate import Evaluator
from engine.model import create_model, load_model
from engine.test import CatBreedPredictor
from engine.trainer import CatModelTrainer
from utils.data_utils import (
    get_multitask_data_loaders_from_json,
    get_single_task_data_loaders_from_json,
)
from utils.parser import (
    create_model_config,
    create_optimizer_and_scheduler,
    create_optimizer_config,
    load_and_update_config,
    parse_args,
    validate_input_path,
    validate_json_datasets,
)
from utils.visualization import setup_logger

# Set up paths
ROOT_DIR = Path(__file__).parent
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT_DIR.parent / "data"))
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Create logger
logger = logging.getLogger(__name__)


def train(args: argparse.Namespace) -> None:
    """Train a model."""
    logger.info(f"Starting training with {args.backbone} backbone in {args.mode} mode")

    if args.mode == "multitask":
        train_multitask(args)
    else:
        train_single_task(args)


def train_single_task(args: argparse.Namespace) -> None:
    """Train a single-task model using JSON datasets."""
    logger.info(f"Loading {args.mode} data from JSON files")

    # Load data and create model
    data_loaders = get_single_task_data_loaders_from_json(
        train_json_path=args.train_json,
        val_json_path=args.val_json,
        test_json_path=args.test_json,
        data_root=args.data_root,
        task=args.mode,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment_data=True,
    )

    train_loader, val_loader, class_names = (
        data_loaders["train"],
        data_loaders["val"],
        data_loaders["class_names"],
    )

    # Create model and training components
    model_config = create_model_config(args)
    model = create_model(
        num_classes=len(class_names), model_config=model_config, model_type=args.mode
    )

    # Configure scheduler parameters from config file
    scheduler_config = {
        "patience": args.scheduler_patience,
        "factor": args.scheduler_factor,
        "min_lr": args.scheduler_min_lr,
        "step_size": args.scheduler_step_size,
        "gamma": args.scheduler_gamma,
        "T_max": getattr(args, "scheduler_t_max", 10),
    }

    # Log optimizer and scheduler configuration from config file
    logger.info(f"Using optimizer configuration from config file: {args.optimizer}")
    logger.info(f"Using scheduler configuration from config file: {args.scheduler}")

    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(
        model,
        args.lr,
        args.weight_decay,
        optimizer_type=args.optimizer,
        scheduler_type=args.scheduler,
        scheduler_config=scheduler_config,
    )

    # Create and run trainer
    trainer = CatModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        use_tensorboard=getattr(args, "use_tensorboard", True),
    )

    history = trainer.fit(
        epochs=args.epochs, early_stopping_patience=args.patience, save_best_only=True
    )
    logger.info(f"Training completed, model saved to {trainer.checkpoint_dir}")


def train_multitask(args: argparse.Namespace) -> None:
    """Train a multitask model using JSON datasets."""
    from engine.multitask_trainer import CatMultitaskTrainer

    logger.info("Loading multitask data from JSON files")

    # Load data and get class information
    data_loaders = get_multitask_data_loaders_from_json(
        train_json_path=args.train_json,
        val_json_path=args.val_json,
        test_json_path=args.test_json,
        data_root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment_data=True,
    )

    num_breed_classes, num_emotion_classes = (
        data_loaders["num_breed_classes"],
        data_loaders["num_emotion_classes"],
    )
    logger.info(
        f"Multitask model: {num_breed_classes} breed classes, {num_emotion_classes} emotion classes"
    )

    # Create model and training components
    model_config = create_model_config(args, shared_features=args.shared_features)
    model = create_model(
        num_classes=num_breed_classes,
        num_emotion_classes=num_emotion_classes,
        model_config=model_config,
        model_type="multitask",
    )

    # Configure scheduler parameters from config file
    scheduler_config = {
        "patience": args.scheduler_patience,
        "factor": args.scheduler_factor,
        "min_lr": args.scheduler_min_lr,
        "step_size": args.scheduler_step_size,
        "gamma": args.scheduler_gamma,
        "T_max": getattr(args, "scheduler_t_max", 10),
    }

    # Log optimizer and scheduler configuration from config file
    logger.info(f"Using optimizer configuration from config file: {args.optimizer}")
    logger.info(f"Using scheduler configuration from config file: {args.scheduler}")

    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(
        model,
        args.lr,
        args.weight_decay,
        optimizer_type=args.optimizer,
        scheduler_type=args.scheduler,
        scheduler_config=scheduler_config,
    )

    # Create and run trainer
    trainer = CatMultitaskTrainer(
        model=model,
        train_loader=data_loaders["train"],
        val_loader=data_loaders["val"],
        optimizer=optimizer,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        breed_weight=args.breed_weight,
        emotion_weight=args.emotion_weight,
        loss_type="focal" if args.mode == "multitask" else "cross_entropy",
        lr_scheduler=scheduler,
        use_tensorboard=getattr(args, "use_tensorboard", True),
    )

    history = trainer.fit(
        epochs=args.epochs, early_stopping_patience=args.patience, save_best_only=True
    )
    logger.info(f"Multitask training completed, models saved to {trainer.checkpoint_dir}")


def load_test_data(args: argparse.Namespace) -> Dict[str, any]:
    """Load test data based on mode (single-task or multitask)."""
    data_loader_kwargs = {
        "train_json_path": args.train_json,
        "val_json_path": args.val_json,
        "test_json_path": args.test_json,
        "data_root": args.data_root,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "augment_data": False,
    }

    if args.mode == "multitask":
        data_loaders = get_multitask_data_loaders_from_json(**data_loader_kwargs)

        # Log what we loaded
        breed_classes = data_loaders.get("breed_classes", [])
        emotion_classes = data_loaders.get("emotion_classes", [])
        logger.info(
            f"Loaded multitask data with {len(breed_classes)} breed classes and {len(emotion_classes)} emotion classes"
        )

        return data_loaders
    else:
        data_loaders = get_single_task_data_loaders_from_json(task=args.mode, **data_loader_kwargs)

        # Log what we loaded
        class_names = data_loaders.get("class_names", [])
        logger.info(f"Loaded {args.mode} data with {len(class_names)} classes")

        return data_loaders


def load_evaluation_model(args: argparse.Namespace, model_path: str, data_loaders: Dict[str, any]):
    """Load model for evaluation based on mode."""
    model_config = {"backbone": args.backbone, "pretrained": False}

    if args.mode == "multitask":
        return load_model(
            path=model_path,
            num_classes=data_loaders["num_breed_classes"],
            num_emotion_classes=data_loaders["num_emotion_classes"],
            model_config=model_config,
            model_type="multitask",
        )
    else:
        return load_model(
            path=model_path,
            num_classes=len(data_loaders["class_names"]),
            model_config=model_config,
        )


def evaluate(args: argparse.Namespace) -> None:
    """Evaluate a model using JSON datasets."""
    model_path, model_dir, class_names = get_model_path_and_classes(args)

    # Load data and model
    data_loaders = load_test_data(args)
    model = load_evaluation_model(args, model_path, data_loaders)

    # Set up evaluation
    output_dir = args.output_dir or Path(args.checkpoint_dir) / "evaluation"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    evaluator = Evaluator(
        model=model, test_loader=data_loaders["test"], device=args.device, output_dir=output_dir
    )

    # Choose evaluation method based on mode
    if args.mode == "multitask":
        logger.info("Evaluating in multitask mode for both breed and emotion tasks")

        # Get breed and emotion class names using our helper function
        breed_classes, emotion_classes = get_multitask_class_names(args, model_dir)

        # Validate class lists
        if not breed_classes or not emotion_classes:
            logger.error("Missing breed or emotion classes for multitask evaluation")
            sys.exit(1)

        logger.info(
            f"Evaluating with {len(breed_classes)} breed classes and {len(emotion_classes)} emotion classes"
        )
        results = evaluator.evaluate_multitask(
            breed_classes=breed_classes, emotion_classes=emotion_classes
        )
    else:
        logger.info(f"Evaluating in single-task mode for {args.mode} task")
        results = evaluator.evaluate(class_names=class_names, task=args.mode)

    # Log key metrics
    if args.mode == "multitask":
        # For multitask, metrics are nested under 'breed' and 'emotion' keys
        for task in ["breed", "emotion"]:
            if task in results:
                task_results = results[task]
                metrics = ["accuracy", "f1_score", "top3_accuracy"]
                logger.info(f"{task.capitalize()} Evaluation Summary:")
                for metric in metrics:
                    if metric in task_results:
                        logger.info(
                            f"  {metric.replace('_', ' ').title()}: {task_results[metric]:.4f}"
                        )

        if "avg_inference_time_ms" in results:
            logger.info(
                f"Average Inference Time: {results['avg_inference_time_ms']:.2f} ms/sample"
            )
    else:
        # For single task, metrics are at the top level
        metrics = ["accuracy", "f1_score", "top3_accuracy"]
        logger.info("Evaluation Summary:")
        for metric in metrics:
            if metric in results:
                logger.info(f"{metric.replace('_', ' ').title()}: {results[metric]:.4f}")

        if "avg_inference_time_ms" in results:
            logger.info(
                f"Average Inference Time: {results['avg_inference_time_ms']:.2f} ms/sample"
            )


def validate_input_path(path: str, file_type: str) -> None:
    """Validate that input path exists."""
    if not path:
        logger.error(f"Input {file_type} path is required")
        sys.exit(1)
    if not os.path.exists(path):
        logger.error(f"{file_type.title()} not found: {path}")
        sys.exit(1)


def create_predictor(args: argparse.Namespace) -> CatBreedPredictor:
    """Create a predictor instance with model and class names."""
    model_path, model_dir, class_names = get_model_path_and_classes(args)
    return CatBreedPredictor(
        model_path=model_path,
        class_names=class_names,
        device=args.device,
        img_size=args.img_size,
    )


def predict(args: argparse.Namespace) -> None:
    """Run prediction on a single image."""
    validate_input_path(args.input, "image")

    predictor = create_predictor(args)
    fig = predictor.predict_and_visualize(args.input)

    if args.output:
        fig.savefig(args.output)
        logger.info(f"Prediction visualization saved to {Path(args.output).resolve()}")
    else:
        import matplotlib.pyplot as plt

        plt.show()


def process_video(args: argparse.Namespace) -> None:
    """Process a video file."""
    validate_input_path(args.input, "video")

    predictor = create_predictor(args)
    output = predictor.process_video(
        video_path=args.input, output_path=args.output, display=args.output is None
    )

    if output:
        logger.info(f"Processed video saved to {Path(output).resolve()}")


def run_webcam(args: argparse.Namespace) -> None:
    """Run prediction on webcam feed."""
    predictor = create_predictor(args)

    try:
        predictor.run_webcam(camera_id=args.camera_id)
    except KeyboardInterrupt:
        logger.info("Webcam feed stopped by user")


def get_model_path_and_classes(args: argparse.Namespace) -> tuple[Path, Path, List[str]]:
    """Get model path and class names from various sources."""
    model_path, model_dir = None, None

    # Handle training config file
    if args.config_path:
        config_dir = Path(args.config_path).parent
        model_path = config_dir / "best_state.pth"
        if not model_path.exists():
            model_path = config_dir / "last_state.pth"
        if model_path.exists():
            model_dir = config_dir
            logger.info(f"Using model from config directory: {os.path.relpath(model_path)}")

    # Handle direct model path
    if not model_path and args.model_path:
        model_path = Path(args.model_path)
        if model_path.exists():
            model_dir = model_path.parent
            logger.info(f"Using specified model: {os.path.relpath(model_path)}")

    # Find latest checkpoint
    if not model_path or not model_dir:
        model_dirs = sorted(
            [
                d
                for d in Path(args.checkpoint_dir).iterdir()
                if d.is_dir() and any(d.glob("*.pth"))
            ],
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )

        if not model_dirs:
            logger.error(f"No model checkpoints found in {args.checkpoint_dir}")
            sys.exit(1)

        model_dir = model_dirs[0]
        model_path = model_dir / "best_state.pth"
        if not model_path.exists():
            model_path = model_dir / "last_state.pth"

        if not model_path.exists():
            logger.error(f"No model checkpoint files found in {os.path.relpath(model_dir)}")
            sys.exit(1)

        logger.info(f"Using latest model: {os.path.relpath(model_path)}")

    # Load class names
    class_names_path = Path(model_dir) / "class_names.json"
    if class_names_path.exists():
        with class_names_path.open("r") as f:
            class_names = json.load(f)
    else:
        class_names = load_class_names_from_json(args)

    return model_path, model_dir, class_names


def load_class_names_from_json(args: argparse.Namespace) -> List[str]:
    """Load class names from JSON dataset as fallback."""
    try:
        with Path(args.train_json).open("r") as f:
            train_data = json.load(f)

        metadata = train_data.get("metadata", {})
        samples = train_data.get("samples", [])

        if args.mode == "multitask":
            class_names = metadata.get("breed_classes") or sorted(
                {s.get("breed_class") for s in samples if "breed_class" in s}
            )
        else:
            class_names = metadata.get("class_names") or sorted(
                {s.get("class_name") for s in samples if "class_name" in s}
            )

        logger.info(f"Loaded class names from JSON dataset: {len(class_names)} classes")
        return class_names

    except Exception as e:
        logger.error(f"Could not load class names from JSON dataset: {e}")
        return []


def get_multitask_class_names(
    args: argparse.Namespace, model_dir: Path
) -> Tuple[List[str], List[str]]:
    """Get breed and emotion class names for multitask evaluation.

    Args:
        args: Command-line arguments
        model_dir: Directory containing the model checkpoint

    Returns:
        Tuple of (breed_class_names, emotion_class_names)
    """
    # Try to load class names from model directory
    breed_classes = []
    emotion_classes = []

    # Look for breed class names
    breed_class_path = model_dir / "class_names.json"
    if breed_class_path.exists():
        try:
            with breed_class_path.open("r") as f:
                breed_classes = json.load(f)
            logger.info(f"Loaded {len(breed_classes)} breed classes from model directory")
        except Exception as e:
            logger.warning(f"Failed to load breed classes from {breed_class_path}: {e}")

    # Look for emotion class names
    emotion_class_path = model_dir / "emotion_class_names.json"
    if emotion_class_path.exists():
        try:
            with emotion_class_path.open("r") as f:
                emotion_classes = json.load(f)
            logger.info(f"Loaded {len(emotion_classes)} emotion classes from model directory")
        except Exception as e:
            logger.warning(f"Failed to load emotion classes from {emotion_class_path}: {e}")

    # If either class list is missing, try to load from JSON dataset
    if not breed_classes or not emotion_classes:
        try:
            with Path(args.train_json).open("r") as f:
                train_data = json.load(f)

            metadata = train_data.get("metadata", {})
            samples = train_data.get("samples", [])

            # Get breed classes if not already loaded
            if not breed_classes:
                breed_classes = metadata.get("breed_classes") or sorted(
                    {s.get("breed_class") for s in samples if "breed_class" in s}
                )
                if breed_classes:
                    logger.info(f"Loaded {len(breed_classes)} breed classes from JSON dataset")

            # Get emotion classes if not already loaded
            if not emotion_classes:
                emotion_classes = metadata.get("emotion_classes") or sorted(
                    {s.get("emotion_class") for s in samples if "emotion_class" in s}
                )
                if emotion_classes:
                    logger.info(f"Loaded {len(emotion_classes)} emotion classes from JSON dataset")

        except Exception as e:
            logger.warning(f"Failed to load class names from JSON dataset: {e}")

    # If emotion classes are still missing, use default emotion classes as fallback
    if not emotion_classes:
        emotion_classes = ["angry", "happy", "neutral", "sad", "surprise"]
        logger.warning(f"Using default emotion classes: {emotion_classes}")

    return breed_classes, emotion_classes


def validate_json_datasets(args: argparse.Namespace) -> None:
    """Validate required JSON dataset files exist after config loading."""
    required_files = [args.train_json, args.val_json, args.test_json]

    if not all(required_files):
        logger.error("JSON dataset files are required: --train-json, --val-json, --test-json")
        logger.error("Provide them via command line or config file")
        sys.exit(1)

    missing_files = [f for f in required_files if not Path(f).exists()]
    if missing_files:
        logger.error(f"JSON dataset files not found: {', '.join(missing_files)}")
        sys.exit(1)

    if not Path(args.data_root).exists():
        logger.error(f"Data root directory not found: {args.data_root}")
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    args = parse_args()
    args = load_and_update_config(args)

    validate_json_datasets(args)
    setup_logger(args.log_file)

    # Log execution info
    logger.info(f"Running command: {args.command} | Mode: {args.mode} | Device: {args.device}")
    logger.info(
        f"Datasets - Train: {args.train_json} | Val: {args.val_json} | Test: {args.test_json}"
    )
    logger.info(f"Data Root: {args.data_root}")

    # Command dispatch
    commands = {
        "train": train,
        "evaluate": evaluate,
        "predict": predict,
        "video": process_video,
        "webcam": run_webcam,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        logger.error(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
