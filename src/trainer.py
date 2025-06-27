"""Training module for cat breed classification."""

import datetime
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from utils.data_utils import print_training_config
from utils.visualization import plot_learning_curves

logger = logging.getLogger(__name__)


class CatModelTrainer:
    """Simplified trainer class for cat classification."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        checkpoint_dir: str = "checkpoints",
        use_tensorboard: bool = True,
    ):
        """Initialize the trainer with default values."""
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer or optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.use_tensorboard = use_tensorboard

        # Set up checkpoint directory
        backbone_name = getattr(model, "_backbone_name", type(model).__name__)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_type = getattr(model, "__type__", "model")

        self.checkpoint_dir = Path(checkpoint_dir) / f"{model_type}_{backbone_name}_{timestamp}"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_model_path = self.checkpoint_dir / "best_state.pth"
        self.last_model_path = self.checkpoint_dir / "last_state.pth"

        # Set up TensorBoard
        if self.use_tensorboard:
            self.tb_log_dir = self.checkpoint_dir / "tensorboard_logs"
            self.tb_log_dir.mkdir(exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.tb_log_dir)
        else:
            self.writer = None

        # Initialize history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "learning_rates": [],
        }

        logger.info(
            f"Trainer initialized: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val samples, device: {device}"
        )

    def _run_epoch(self, data_loader: DataLoader, training: bool = True) -> Tuple[float, float]:
        """Run one epoch of training or validation."""
        self.model.train(training)
        total_loss, correct, total = 0.0, 0, 0

        desc = "Training" if training else "Validating"
        pbar = tqdm(data_loader, desc=desc)

        with torch.set_grad_enabled(training):
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                if training:
                    self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                if training:
                    loss.backward()
                    self.optimizer.step()

                # Update metrics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # Update progress bar
                pbar.set_postfix(
                    {"loss": total_loss / (pbar.n + 1), "acc": 100.0 * correct / total}
                )

        return total_loss / len(data_loader), 100.0 * correct / total

    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        return self._run_epoch(self.train_loader, training=True)

    def validate(self) -> Tuple[float, float]:
        """Validate the model."""
        return self._run_epoch(self.val_loader, training=False)

    def fit(
        self, epochs: int, early_stopping_patience: int = 10, save_best_only: bool = True
    ) -> Dict[str, List[float]]:
        """Train the model for multiple epochs with simplified logic."""
        best_val_loss, patience_counter = float("inf"), 0

        # Save training config
        self._save_training_config(epochs, early_stopping_patience)
        self._setup_tensorboard()

        logger.info(f"Starting training for {epochs} epochs")

        for epoch in range(1, epochs + 1):
            # Training and validation
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            # Update learning rate and history
            if self.lr_scheduler:
                self.lr_scheduler.step(val_loss)

            current_lr = self.optimizer.param_groups[0]["lr"]
            self._update_history(train_loss, train_acc, val_loss, val_acc, current_lr)
            self._log_tensorboard_metrics(
                epoch, train_loss, train_acc, val_loss, val_acc, current_lr
            )

            # Log epoch results
            logger.info(
                f"Epoch {epoch}/{epochs} - train_loss: {train_loss:.4f} - train_acc: {train_acc:.2f}% - "
                f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.2f}% - lr: {current_lr:.6f}"
            )

            # Save checkpoints and handle early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint(self.best_model_path, epoch, val_loss, val_acc, is_best=True)
                logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
            else:
                patience_counter += 1

            self.save_checkpoint(self.last_model_path, epoch, val_loss, val_acc, is_best=False)

            # Early stopping check
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break

        self._finalize_training()
        return self.history

    def _save_training_config(self, epochs: int, patience: int) -> None:
        """Save training configuration to YAML file."""
        training_config = {
            "backbone": getattr(self.model, "_backbone_name", "unknown"),
            "epochs": epochs,
            "batch_size": self.train_loader.batch_size,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "weight_decay": self.optimizer.param_groups[0].get("weight_decay", 0),
            "early_stopping": patience,
            "device": self.device,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        config_path = self.checkpoint_dir / "training_config.yaml"
        with config_path.open("w") as f:
            yaml.dump(training_config, f, default_flow_style=False)

        print_training_config(training_config)

    def _setup_tensorboard(self) -> None:
        """Set up TensorBoard logging with sample images and model graph."""
        if not (self.use_tensorboard and self.writer):
            return

        try:
            import torchvision

            sample_batch, _ = next(iter(self.train_loader))
            sample_batch = sample_batch[:8]

            # Add model graph (CPU copy to avoid CUDA issues)
            temp_model = type(self.model)()
            temp_model.load_state_dict(self.model.state_dict())
            self.writer.add_graph(temp_model.cpu(), sample_batch.cpu())

            # Add sample images
            grid = torchvision.utils.make_grid(sample_batch, normalize=True)
            self.writer.add_image("Sample training images", grid, 0)
        except Exception as e:
            logger.warning(f"Could not set up TensorBoard visualization: {e}")

    def _update_history(
        self, train_loss: float, train_acc: float, val_loss: float, val_acc: float, lr: float
    ) -> None:
        """Update training history."""
        metrics = [train_loss, train_acc, val_loss, val_acc, lr]
        keys = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rates"]

        for key, value in zip(keys, metrics):
            self.history[key].append(value)

    def _log_tensorboard_metrics(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        lr: float,
    ) -> None:
        """Log metrics to TensorBoard."""
        if not (self.use_tensorboard and self.writer):
            return

        metrics = {
            "Loss/train": train_loss,
            "Loss/val": val_loss,
            "Accuracy/train": train_acc,
            "Accuracy/val": val_acc,
            "Learning_rate": lr,
        }

        for name, value in metrics.items():
            self.writer.add_scalar(name, value, epoch)

    def _finalize_training(self) -> None:
        """Finalize training by saving artifacts and closing resources."""
        # Save learning curves
        fig = plot_learning_curves(self.history)
        fig.savefig(self.checkpoint_dir / "learning_curves.png")

        # Save class names if available
        if hasattr(self.train_loader.dataset, "class_names"):
            import json

            with (self.checkpoint_dir / "class_names.json").open("w") as f:
                json.dump(self.train_loader.dataset.class_names, f)

        # Close TensorBoard
        if self.writer:
            self.writer.close()
            logger.info(f"TensorBoard logs saved to {self.tb_log_dir.relative_to(Path.cwd())}")

        logger.info(f"Training artifacts saved to {self.checkpoint_dir.relative_to(Path.cwd())}")

    def save_checkpoint(
        self, path: Path, epoch: int, val_loss: float, val_acc: float, is_best: bool = False
    ) -> None:
        """Save a model checkpoint with essential information."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "history": self.history,
        }

        if self.lr_scheduler:
            checkpoint["scheduler_state_dict"] = self.lr_scheduler.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Optional[Path] = None) -> int:
        """Load a model checkpoint and return the epoch number."""
        if path is None:
            path = self.best_model_path if self.best_model_path.exists() else self.last_model_path

        if not path.exists():
            logger.error(f"Checkpoint file not found: {path}")
            return 0

        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint and self.lr_scheduler:
            self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if "history" in checkpoint:
            self.history = checkpoint["history"]

        epoch = checkpoint.get("epoch", 0)
        logger.info(f"Loaded checkpoint from {path.relative_to(Path.cwd())} (epoch {epoch})")
        return epoch
