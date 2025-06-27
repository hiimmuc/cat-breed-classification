"""Multitask trainer for joint breed and emotion classification."""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from utils.data_utils import get_multitask_data_loaders

logger = logging.getLogger(__name__)


class MultitaskLoss(nn.Module):
    """Combined loss for multitask learning with task weighting."""

    def __init__(
        self,
        breed_weight: float = 1.0,
        emotion_weight: float = 1.0,
        loss_type: str = "cross_entropy",
    ):
        """
        Initialize multitask loss.

        Args:
            breed_weight: Weight for breed classification loss
            emotion_weight: Weight for emotion classification loss
            loss_type: Type of loss function ("cross_entropy", "focal")
        """
        super().__init__()
        self.breed_weight = breed_weight
        self.emotion_weight = emotion_weight

        if loss_type == "cross_entropy":
            self.breed_loss_fn = nn.CrossEntropyLoss()
            self.emotion_loss_fn = nn.CrossEntropyLoss()
        elif loss_type == "focal":
            # For now, use cross entropy - can implement focal loss later if needed
            logger.warning("Focal loss not implemented, using cross entropy instead")
            self.breed_loss_fn = nn.CrossEntropyLoss()
            self.emotion_loss_fn = nn.CrossEntropyLoss()
            self.emotion_loss_fn = FocalLoss(alpha=1, gamma=2)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def forward(
        self,
        breed_logits: torch.Tensor,
        emotion_logits: torch.Tensor,
        breed_targets: torch.Tensor,
        emotion_targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss.

        Args:
            breed_logits: Breed prediction logits
            emotion_logits: Emotion prediction logits
            breed_targets: Breed ground truth labels
            emotion_targets: Emotion ground truth labels

        Returns:
            Tuple of (combined_loss, individual_losses_dict)
        """
        breed_loss = self.breed_loss_fn(breed_logits, breed_targets)
        emotion_loss = self.emotion_loss_fn(emotion_logits, emotion_targets)

        combined_loss = self.breed_weight * breed_loss + self.emotion_weight * emotion_loss

        losses = {
            "breed_loss": breed_loss,
            "emotion_loss": emotion_loss,
            "combined_loss": combined_loss,
        }

        return combined_loss, losses


class CatMultitaskTrainer:
    """Trainer for multitask cat breed and emotion classification."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
        checkpoint_dir: str = "checkpoints",
        breed_weight: float = 1.0,
        emotion_weight: float = 1.0,
        loss_type: str = "cross_entropy",
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        use_tensorboard: bool = True,
    ):
        """
        Initialize multitask trainer.

        Args:
            model: Multitask model
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            device: Device to use
            checkpoint_dir: Directory to save checkpoints
            breed_weight: Weight for breed classification loss
            emotion_weight: Weight for emotion classification loss
            loss_type: Type of loss function
            lr_scheduler: Learning rate scheduler
            use_tensorboard: Whether to use tensorboard logging
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.lr_scheduler = lr_scheduler

        # Set up checkpoint directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.checkpoint_dir = (
            Path(checkpoint_dir) / f"multitask_{model._backbone_name}_{timestamp}"
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Set up loss function
        self.loss_fn = MultitaskLoss(
            breed_weight=breed_weight, emotion_weight=emotion_weight, loss_type=loss_type
        ).to(device)

        # Set up tensorboard
        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            self.writer = SummaryWriter(log_dir=self.checkpoint_dir / "tensorboard_logs")

        # Training history
        self.history = {
            "train_loss": [],
            "train_breed_loss": [],
            "train_emotion_loss": [],
            "train_breed_acc": [],
            "train_emotion_acc": [],
            "val_loss": [],
            "val_breed_loss": [],
            "val_emotion_loss": [],
            "val_breed_acc": [],
            "val_emotion_acc": [],
            "lr": [],
        }

        logger.info(
            f"Multitask trainer initialized. Checkpoints will be saved to {self.checkpoint_dir}"
        )

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_breed_loss = 0.0
        total_emotion_loss = 0.0
        total_breed_correct = 0
        total_emotion_correct = 0
        total_samples = 0

        # Create progress bar
        pbar = tqdm(self.train_loader, desc="Training")

        for batch_idx, batch in enumerate(pbar):
            # Unpack batch - expecting (images, breed_labels, emotion_labels)
            if len(batch) == 3:
                images, breed_labels, emotion_labels = batch
            else:
                # Fallback for single-task data loaders
                images, labels = batch
                breed_labels = labels
                emotion_labels = labels  # This won't be ideal, but allows compatibility

            images = images.to(self.device)
            breed_labels = breed_labels.to(self.device)
            emotion_labels = emotion_labels.to(self.device)

            # Forward pass
            breed_logits, emotion_logits = self.model(images, task="both")

            # Compute loss
            loss, losses = self.loss_fn(breed_logits, emotion_logits, breed_labels, emotion_labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            total_breed_loss += losses["breed_loss"].item()
            total_emotion_loss += losses["emotion_loss"].item()

            # Accuracy
            breed_pred = torch.argmax(breed_logits, dim=1)
            emotion_pred = torch.argmax(emotion_logits, dim=1)

            total_breed_correct += (breed_pred == breed_labels).sum().item()
            total_emotion_correct += (emotion_pred == emotion_labels).sum().item()
            total_samples += images.size(0)

            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            breed_acc = total_breed_correct / total_samples * 100
            emotion_acc = total_emotion_correct / total_samples * 100

            pbar.set_postfix(
                {
                    "loss": f"{avg_loss:.4f}",
                    "breed_acc": f"{breed_acc:.1f}%",
                    "emotion_acc": f"{emotion_acc:.1f}%",
                }
            )

            # Log batch-level metrics
            if self.use_tensorboard and batch_idx % 50 == 0:
                step = len(self.train_loader) * (len(self.history["train_loss"])) + batch_idx
                self.writer.add_scalar("Batch/Train_Loss", loss.item(), step)
                self.writer.add_scalar("Batch/Train_Breed_Loss", losses["breed_loss"].item(), step)
                self.writer.add_scalar(
                    "Batch/Train_Emotion_Loss", losses["emotion_loss"].item(), step
                )

        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        avg_breed_loss = total_breed_loss / len(self.train_loader)
        avg_emotion_loss = total_emotion_loss / len(self.train_loader)
        breed_acc = total_breed_correct / total_samples
        emotion_acc = total_emotion_correct / total_samples

        return {
            "loss": avg_loss,
            "breed_loss": avg_breed_loss,
            "emotion_loss": avg_emotion_loss,
            "breed_acc": breed_acc,
            "emotion_acc": emotion_acc,
        }

    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()

        total_loss = 0.0
        total_breed_loss = 0.0
        total_emotion_loss = 0.0
        total_breed_correct = 0
        total_emotion_correct = 0
        total_samples = 0

        # Create progress bar
        pbar = tqdm(self.val_loader, desc="Validating")

        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                # Unpack batch
                if len(batch) == 3:
                    images, breed_labels, emotion_labels = batch
                else:
                    images, labels = batch
                    breed_labels = labels
                    emotion_labels = labels

                images = images.to(self.device)
                breed_labels = breed_labels.to(self.device)
                emotion_labels = emotion_labels.to(self.device)

                # Forward pass
                breed_logits, emotion_logits = self.model(images, task="both")

                # Compute loss
                loss, losses = self.loss_fn(
                    breed_logits, emotion_logits, breed_labels, emotion_labels
                )

                # Statistics
                total_loss += loss.item()
                total_breed_loss += losses["breed_loss"].item()
                total_emotion_loss += losses["emotion_loss"].item()

                # Accuracy
                breed_pred = torch.argmax(breed_logits, dim=1)
                emotion_pred = torch.argmax(emotion_logits, dim=1)

                total_breed_correct += (breed_pred == breed_labels).sum().item()
                total_emotion_correct += (emotion_pred == emotion_labels).sum().item()
                total_samples += images.size(0)

                # Update progress bar
                avg_loss = total_loss / (batch_idx + 1)
                breed_acc = total_breed_correct / total_samples * 100
                emotion_acc = total_emotion_correct / total_samples * 100

                pbar.set_postfix(
                    {
                        "loss": f"{avg_loss:.4f}",
                        "breed_acc": f"{breed_acc:.1f}%",
                        "emotion_acc": f"{emotion_acc:.1f}%",
                    }
                )

        # Calculate epoch metrics
        avg_loss = total_loss / len(self.val_loader)
        avg_breed_loss = total_breed_loss / len(self.val_loader)
        avg_emotion_loss = total_emotion_loss / len(self.val_loader)
        breed_acc = total_breed_correct / total_samples
        emotion_acc = total_emotion_correct / total_samples

        return {
            "loss": avg_loss,
            "breed_loss": avg_breed_loss,
            "emotion_loss": avg_emotion_loss,
            "breed_acc": breed_acc,
            "emotion_acc": emotion_acc,
        }

    def fit(
        self,
        epochs: int,
        early_stopping_patience: int = 10,
        save_best_only: bool = True,
        verbose: bool = True,
    ) -> Dict:
        """
        Train the multitask model.

        Args:
            epochs: Number of epochs to train
            early_stopping_patience: Patience for early stopping
            save_best_only: Whether to save only the best model
            verbose: Whether to print training progress

        Returns:
            Training history dictionary
        """
        best_val_loss = float("inf")
        patience_counter = 0

        logger.info(f"Starting multitask training for {epochs} epochs")

        # Create epoch progress bar
        epoch_pbar = tqdm(range(epochs), desc="Epochs", position=0)

        for epoch in epoch_pbar:
            epoch_start_time = time.time()

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate_epoch()

            # Update learning rate
            if self.lr_scheduler:
                if hasattr(self.lr_scheduler, "step"):
                    if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.lr_scheduler.step(val_metrics["loss"])
                    else:
                        self.lr_scheduler.step()

            # Record history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_breed_loss"].append(train_metrics["breed_loss"])
            self.history["train_emotion_loss"].append(train_metrics["emotion_loss"])
            self.history["train_breed_acc"].append(train_metrics["breed_acc"])
            self.history["train_emotion_acc"].append(train_metrics["emotion_acc"])

            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_breed_loss"].append(val_metrics["breed_loss"])
            self.history["val_emotion_loss"].append(val_metrics["emotion_loss"])
            self.history["val_breed_acc"].append(val_metrics["breed_acc"])
            self.history["val_emotion_acc"].append(val_metrics["emotion_acc"])

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["lr"].append(current_lr)

            # Tensorboard logging
            if self.use_tensorboard:
                self.writer.add_scalar("Epoch/Train_Loss", train_metrics["loss"], epoch)
                self.writer.add_scalar(
                    "Epoch/Train_Breed_Loss", train_metrics["breed_loss"], epoch
                )
                self.writer.add_scalar(
                    "Epoch/Train_Emotion_Loss", train_metrics["emotion_loss"], epoch
                )
                self.writer.add_scalar(
                    "Epoch/Train_Breed_Accuracy", train_metrics["breed_acc"], epoch
                )
                self.writer.add_scalar(
                    "Epoch/Train_Emotion_Accuracy", train_metrics["emotion_acc"], epoch
                )

                self.writer.add_scalar("Epoch/Val_Loss", val_metrics["loss"], epoch)
                self.writer.add_scalar("Epoch/Val_Breed_Loss", val_metrics["breed_loss"], epoch)
                self.writer.add_scalar(
                    "Epoch/Val_Emotion_Loss", val_metrics["emotion_loss"], epoch
                )
                self.writer.add_scalar("Epoch/Val_Breed_Accuracy", val_metrics["breed_acc"], epoch)
                self.writer.add_scalar(
                    "Epoch/Val_Emotion_Accuracy", val_metrics["emotion_acc"], epoch
                )

                self.writer.add_scalar("Epoch/Learning_Rate", current_lr, epoch)

            # Print progress
            if verbose:
                epoch_time = time.time() - epoch_start_time
                logger.info(
                    f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s) - "
                    f"Train Loss: {train_metrics['loss']:.4f} "
                    f"(Breed: {train_metrics['breed_loss']:.4f}, Emotion: {train_metrics['emotion_loss']:.4f}) - "
                    f"Train Acc: {train_metrics['breed_acc']:.4f}/{train_metrics['emotion_acc']:.4f} - "
                    f"Val Loss: {val_metrics['loss']:.4f} "
                    f"(Breed: {val_metrics['breed_loss']:.4f}, Emotion: {val_metrics['emotion_loss']:.4f}) - "
                    f"Val Acc: {val_metrics['breed_acc']:.4f}/{val_metrics['emotion_acc']:.4f} - "
                    f"LR: {current_lr:.2e}"
                )

            # Update epoch progress bar
            epoch_pbar.set_postfix(
                {
                    "train_loss": f"{train_metrics['loss']:.4f}",
                    "val_loss": f"{val_metrics['loss']:.4f}",
                    "breed_acc": f"{val_metrics['breed_acc']:.3f}",
                    "emotion_acc": f"{val_metrics['emotion_acc']:.3f}",
                    "patience": f"{patience_counter}/{early_stopping_patience}",
                }
            )

            # Save best model
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0

                if save_best_only:
                    self.save_checkpoint(
                        epoch=epoch,
                        loss=val_metrics["loss"],
                        breed_accuracy=val_metrics["breed_acc"],
                        emotion_accuracy=val_metrics["emotion_acc"],
                        filename="best_state.pth",
                    )
            else:
                patience_counter += 1

            # Save last model
            self.save_checkpoint(
                epoch=epoch,
                loss=val_metrics["loss"],
                breed_accuracy=val_metrics["breed_acc"],
                emotion_accuracy=val_metrics["emotion_acc"],
                filename="last_state.pth",
            )

            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

        # Close epoch progress bar
        epoch_pbar.close()

        # Save training curves
        self.plot_training_curves()

        if self.use_tensorboard:
            self.writer.close()

        logger.info("Training completed!")

        return self.history

    def save_checkpoint(
        self,
        epoch: int,
        loss: float,
        breed_accuracy: float,
        emotion_accuracy: float,
        filename: str = "checkpoint.pth",
    ):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss,
            "breed_accuracy": breed_accuracy,
            "emotion_accuracy": emotion_accuracy,
            "history": self.history,
        }

        if self.lr_scheduler:
            checkpoint["scheduler_state_dict"] = self.lr_scheduler.state_dict()

        torch.save(checkpoint, self.checkpoint_dir / filename)

    def plot_training_curves(self):
        """Plot and save training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        epochs = range(1, len(self.history["train_loss"]) + 1)

        # Loss curves
        axes[0, 0].plot(epochs, self.history["train_loss"], "b-", label="Train")
        axes[0, 0].plot(epochs, self.history["val_loss"], "r-", label="Validation")
        axes[0, 0].set_title("Combined Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Task-specific losses
        axes[0, 1].plot(epochs, self.history["train_breed_loss"], "b-", label="Train Breed")
        axes[0, 1].plot(epochs, self.history["val_breed_loss"], "b--", label="Val Breed")
        axes[0, 1].plot(epochs, self.history["train_emotion_loss"], "r-", label="Train Emotion")
        axes[0, 1].plot(epochs, self.history["val_emotion_loss"], "r--", label="Val Emotion")
        axes[0, 1].set_title("Task-specific Losses")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Breed accuracy
        axes[1, 0].plot(epochs, self.history["train_breed_acc"], "b-", label="Train")
        axes[1, 0].plot(epochs, self.history["val_breed_acc"], "r-", label="Validation")
        axes[1, 0].set_title("Breed Classification Accuracy")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Accuracy")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Emotion accuracy
        axes[1, 1].plot(epochs, self.history["train_emotion_acc"], "b-", label="Train")
        axes[1, 1].plot(epochs, self.history["val_emotion_acc"], "r-", label="Validation")
        axes[1, 1].set_title("Emotion Classification Accuracy")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Accuracy")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(self.checkpoint_dir / "learning_curves.png", dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Training curves saved to {self.checkpoint_dir / 'learning_curves.png'}")
