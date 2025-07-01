"""Command line argument parsing and configuration handling."""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict

import torch
import yaml

logger = logging.getLogger(__name__)

# Set up paths
ROOT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT_DIR.parent / "data"))


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
    # Add default optimizer and scheduler attributes to args
    optimizer_defaults = {
        "optimizer": "adamw",
        "scheduler": "reduce_lr_on_plateau",
        "scheduler_patience": 3,
        "scheduler_factor": 0.5,
        "scheduler_min_lr": 1e-6,
        "scheduler_step_size": 10,
        "scheduler_gamma": 0.1,
        "scheduler_t_max": 10,
    }

    # Set optimizer and scheduler defaults (these will be overridden by config file)
    for attr, value in optimizer_defaults.items():
        if not hasattr(args, attr):
            setattr(args, attr, value)

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
            # Optimizer and scheduler configs
            "optimizer": ("optimizer", "adamw", str),
            "scheduler": ("scheduler", "reduce_lr_on_plateau", str),
            "scheduler_patience": ("scheduler_patience", 3, int),
            "scheduler_factor": ("scheduler_factor", 0.5, float),
            "scheduler_min_lr": ("scheduler_min_lr", 1e-6, float),
            "scheduler_step_size": ("scheduler_step_size", 10, int),
            "scheduler_gamma": ("scheduler_gamma", 0.1, float),
            "scheduler_t_max": ("scheduler_t_max", 10, int),
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


def create_optimizer_config(args: argparse.Namespace) -> Dict[str, any]:
    """Create optimizer configuration dictionary from parsed arguments."""
    scheduler_config = {
        "patience": args.scheduler_patience,
        "factor": args.scheduler_factor,
        "min_lr": args.scheduler_min_lr,
        "step_size": args.scheduler_step_size,
        "gamma": args.scheduler_gamma,
        "T_max": getattr(args, "scheduler_t_max", 10),
    }

    return {
        "optimizer_type": args.optimizer,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "scheduler_type": args.scheduler,
        "scheduler_config": scheduler_config,
    }


def create_model_config(args: argparse.Namespace, shared_features: bool = False) -> Dict[str, any]:
    """Create model configuration dictionary."""
    return {
        "backbone": args.backbone,
        "pretrained": True,
        "shared_features": shared_features,
    }


def validate_input_path(path: str, file_type: str) -> None:
    """Validate that input path exists."""
    if not path:
        logger.error(f"Input {file_type} path is required")
        sys.exit(1)
    if not os.path.exists(path):
        logger.error(f"{file_type.title()} not found: {path}")
        sys.exit(1)


def create_optimizer_and_scheduler(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
    optimizer_type: str = "adamw",
    scheduler_type: str = "reduce_lr_on_plateau",
    scheduler_config: Dict[str, any] = None,
) -> tuple:
    """Create optimizer and learning rate scheduler.

    Args:
        model: The model to optimize.
        lr: Learning rate.
        weight_decay: Weight decay regularization.
        optimizer_type: Type of optimizer ('adamw', 'adam', 'sgd').
        scheduler_type: Type of scheduler ('reduce_lr_on_plateau', 'cosine', 'step', 'none').
        scheduler_config: Dictionary of scheduler configuration parameters.

    Returns:
        Tuple of (optimizer, scheduler)
    """
    # Set default scheduler config
    if scheduler_config is None:
        scheduler_config = {
            "patience": 3,
            "factor": 0.5,
            "min_lr": 1e-6,
            "step_size": 10,
            "gamma": 0.1,
            "T_max": 10,
        }

    # Create optimizer based on type
    optimizer_type = optimizer_type.lower()
    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9
        )
    else:  # default to AdamW
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    logger.info(f"Using optimizer: {optimizer.__class__.__name__}")

    # Create scheduler based on type
    scheduler_type = scheduler_type.lower()
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config.get("T_max", 10),
            eta_min=scheduler_config.get("min_lr", 1e-6),
        )
        logger.info(f"Using scheduler: CosineAnnealingLR")
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get("step_size", 10),
            gamma=scheduler_config.get("gamma", 0.1),
        )
        logger.info(
            f"Using scheduler: StepLR with step_size={scheduler_config.get('step_size', 10)}"
        )
    elif scheduler_type == "none":
        scheduler = None
        logger.info("No learning rate scheduler selected")
    else:  # default to ReduceLROnPlateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_config.get("factor", 0.5),
            patience=scheduler_config.get("patience", 3),
            min_lr=scheduler_config.get("min_lr", 1e-6),
        )
        logger.info(
            f"Using scheduler: ReduceLROnPlateau with patience={scheduler_config.get('patience', 3)}"
        )

    return optimizer, scheduler
