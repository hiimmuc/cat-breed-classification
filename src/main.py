"""Main entry point for cat breed classification."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.optim as optim
import yaml

HOME = Path(__file__).resolve().parent

# Import local modules
sys.path.insert(0, str(HOME))
import test
from test import CatBreedPredictor

from evaluate import Evaluator
from model import create_model, load_model
from trainer import CatModelTrainer
from utils.data_utils import (
    get_multitask_data_loaders_from_json,
    get_single_task_data_loaders_from_json,
)
from utils.visualization import setup_logger

# Set up paths
ROOT_DIR = Path(__file__).parent
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT_DIR.parent / "data"))
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Create logger
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Cat Classification Pipeline")

    parser.add_argument("command", choices=["train", "evaluate", "predict", "video", "webcam"])
    parser.add_argument("--config-path", type=str, help="YAML config file path")

    # Core arguments
    parser.add_argument("--data-root", type=str, default=str(DATA_DIR))
    parser.add_argument("--checkpoint-dir", type=str, default=str(CHECKPOINT_DIR))
    parser.add_argument("--model-path", type=str, help="Model checkpoint path")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--mode", type=str, choices=["breed", "emotion", "multitask"], default="breed"
    )

    # Model config
    parser.add_argument("--backbone", type=str, default="mobilenet_v2")
    parser.add_argument("--img-size", type=int, default=224)

    # Dataset paths (can be provided via config file)
    parser.add_argument("--train-json", type=str, help="Path to training JSON dataset file")
    parser.add_argument("--val-json", type=str, help="Path to validation JSON dataset file")
    parser.add_argument("--test-json", type=str, help="Path to test JSON dataset file")

    # Training params
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)

    # Multitask weights
    parser.add_argument("--breed-weight", type=float, default=1.0)
    parser.add_argument("--emotion-weight", type=float, default=1.0)
    parser.add_argument("--shared-features", action="store_true", default=False)

    # I/O
    parser.add_argument("--input", type=str, help="Input image/video path")
    parser.add_argument("--output", type=str, help="Output path")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--log-file", type=str, help="Log file path")

    return parser.parse_args()


def load_and_update_config(args) -> argparse.Namespace:
    """Load configuration from YAML file and update args with CLI precedence."""
    if not args.config_path:
        return args

    config_path = Path(args.config_path)
    if not config_path.exists():
        logger.warning(f"Configuration file not found: {config_path}")
        return args

    try:
        logger.info(f"Loading configuration from {config_path}")
        with config_path.open("r") as f:
            config = yaml.safe_load(f)

        if not config:
            logger.warning(f"Empty or invalid configuration file: {config_path}")
            return args

        # Handle special case for training_config.yaml during evaluation
        if "training_config.yaml" in str(config_path) and args.command == "evaluate":
            if not args.checkpoint_dir or args.checkpoint_dir == str(CHECKPOINT_DIR):
                args.checkpoint_dir = str(config_path.parent)
                logger.info(
                    f"Using training config directory as checkpoint_dir: {config_path.parent}"
                )

        # Configuration mapping with type conversion info
        config_mapping = {
            # Basic config -> (arg_name, default_value, type_converter)
            "backbone": ("backbone", "mobilenet_v2", str),
            "batch_size": ("batch_size", 32, int),
            "learning_rate": ("lr", 3e-4, float),
            "weight_decay": ("weight_decay", 1e-4, float),
            "epochs": ("epochs", 50, int),
            "early_stopping": ("patience", 10, int),
            "device": ("device", "cuda" if torch.cuda.is_available() else "cpu", str),
            "img_size": ("img_size", 224, int),
            "num_workers": ("num_workers", 4, int),
            "checkpoint_dir": ("checkpoint_dir", str(CHECKPOINT_DIR), str),
            "data_root": ("data_root", str(DATA_DIR), str),
            "output_dir": ("output_dir", None, str),
            "model_path": ("model_path", None, str),
            "mode": ("mode", "breed", str),
            "breed_weight": ("breed_weight", 1.0, float),
            "emotion_weight": ("emotion_weight", 1.0, float),
            "shared_features": ("shared_features", True, bool),
            "train_json": ("train_json", None, str),
            "val_json": ("val_json", None, str),
            "test_json": ("test_json", None, str),
        }

        # Apply config values only if args use defaults (CLI takes precedence)
        for config_key, (arg_name, default_value, type_converter) in config_mapping.items():
            if hasattr(args, arg_name) and config_key in config:
                current_value = getattr(args, arg_name)
                if current_value == default_value:
                    config_value = type_converter(config[config_key])
                    setattr(args, arg_name, config_value)
                    logger.debug(f"Applied config: {arg_name} = {config_value}")
                elif current_value != default_value:
                    logger.debug(f"Keeping CLI argument: {arg_name} = {current_value}")

        logger.info("Configuration loaded successfully")
        return args

    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return args


def create_optimizer_and_scheduler(
    model: torch.nn.Module, lr: float, weight_decay: float
) -> tuple:
    """Create optimizer and learning rate scheduler."""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.2, patience=20, min_lr=1e-6
    )
    return optimizer, scheduler


def create_model_config(args: argparse.Namespace, shared_features: bool = False) -> Dict[str, any]:
    """Create model configuration dictionary."""
    return {
        "backbone": args.backbone,
        "pretrained": True,
        "shared_features": shared_features,
    }


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
    optimizer, scheduler = create_optimizer_and_scheduler(model, args.lr, args.weight_decay)

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
    from multitask_trainer import CatMultitaskTrainer

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
    optimizer, scheduler = create_optimizer_and_scheduler(model, args.lr, args.weight_decay)

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
        loss_type="cross_entropy",
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
        return get_multitask_data_loaders_from_json(**data_loader_kwargs)
    else:
        return get_single_task_data_loaders_from_json(task=args.mode, **data_loader_kwargs)


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
    results = evaluator.evaluate(class_names=class_names)

    # Log key metrics
    metrics = ["accuracy", "f1_score", "top3_accuracy"]
    logger.info("Evaluation Summary:")
    for metric in metrics:
        if metric in results:
            logger.info(f"{metric.replace('_', ' ').title()}: {results[metric]:.4f}")

    if "avg_inference_time_ms" in results:
        logger.info(f"Average Inference Time: {results['avg_inference_time_ms']:.2f} ms/sample")


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
        print(model_path)
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
            logger.error(f"No model checkpoint files found in {model_dir.relative_to(Path.cwd())}")
            sys.exit(1)

        logger.info(f"Using latest model: {os.path.relpath(model_path)}")
    print(model_dir)
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
