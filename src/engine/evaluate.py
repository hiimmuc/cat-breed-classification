"""
Evaluation module for cat breed and emotion classification with multitask support.

This module provides comprehensive evaluation capabilities for both single-task and
multitask models, including detailed metrics, confusion matrices, and result visualization.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.visualization import plot_confusion_matrix, print_classification_report

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluator class for single-task and multitask cat classification."""

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: str = "cuda",
        output_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the evaluator.

        Args:
            model: Model to evaluate
            test_loader: Test data loader
            device: Device to use for evaluation
            output_dir: Directory to save results
        """
        self.model = model
        self.test_loader = test_loader
        self.device = device if torch.cuda.is_available() else "cpu"

        # Handle output directory path
        if output_dir:
            self.output_dir = Path(output_dir).resolve()
        else:
            self.output_dir = None

        # Move model to device
        self.model.to(self.device)
        self.model.eval()

        if self.output_dir:
            self.output_dir.mkdir(exist_ok=True, parents=True)

        logger.info(
            f"Evaluator initialized with {len(test_loader.dataset)} test samples "
            f"using device: {self.device}"
        )

    def evaluate(self, class_names: List[str], task: str = "breed") -> Dict[str, Any]:
        """
        Evaluate the model for a single task.

        Args:
            class_names: List of class names
            task: Task to evaluate ("breed" or "emotion")

        Returns:
            Dictionary containing evaluation metrics
        """
        all_targets = []
        all_predictions = []
        all_probabilities = []
        total_time = 0
        num_samples = len(self.test_loader.dataset)

        logger.info(f"Starting evaluation on {num_samples} samples for {task} task")

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc=f"Evaluating {task}"):
                # Handle different batch formats
                if len(batch) == 3:
                    # Multitask format: (inputs, breed_targets, emotion_targets)
                    inputs, breed_targets, emotion_targets = batch
                    targets = breed_targets if task == "breed" else emotion_targets
                elif len(batch) == 2:
                    # Single task format: (inputs, targets)
                    inputs, targets = batch
                else:
                    raise ValueError(f"Unexpected batch format with {len(batch)} elements")

                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Measure inference time
                start_time = time.time()

                # Handle different model types
                try:
                    # Try multitask model first
                    if hasattr(self.model, "forward") and task in ["breed", "emotion"]:
                        outputs = self.model(inputs, task=task)
                    else:
                        outputs = self.model(inputs)
                except TypeError:
                    # Fallback to regular forward pass
                    outputs = self.model(inputs)

                batch_time = time.time() - start_time
                total_time += batch_time

                # Get predictions and probabilities
                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = outputs.max(1)

                # Collect results
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        # Convert to numpy arrays
        all_targets = np.array(all_targets)
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)

        # Calculate metrics
        return self._calculate_task_metrics(
            all_targets,
            all_predictions,
            all_probabilities,
            class_names,
            task,
            total_time / num_samples,
        )

    def _calculate_topk_accuracy(
        self, probabilities: np.ndarray, targets: np.ndarray, k: int = 5
    ) -> float:
        """
        Calculate top-k accuracy.

        Args:
            probabilities: Prediction probabilities
            targets: True labels
            k: k value for top-k accuracy

        Returns:
            Top-k accuracy
        """
        batch_size = targets.shape[0]
        top_k_predictions = np.argsort(-probabilities, axis=1)[:, :k]

        # Check if target is in top-k predictions for each sample
        correct = 0
        for i in range(batch_size):
            if targets[i] in top_k_predictions[i]:
                correct += 1

        return correct / batch_size

    def _calculate_per_class_metrics(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        class_names: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate per-class metrics with robust error handling.

        Args:
            targets: True labels
            predictions: Predicted labels
            probabilities: Prediction probabilities
            class_names: List of class names

        Returns:
            Dictionary of per-class metrics
        """
        try:
            # Calculate precision, recall, f1 per class
            precision, recall, f1, support = precision_recall_fscore_support(
                targets, predictions, average=None, zero_division=0
            )

            # Ensure we have the right number of classes
            num_classes = len(class_names)
            if len(precision) != num_classes:
                logger.warning(
                    f"Mismatch between number of classes ({num_classes}) and metrics ({len(precision)})"
                )
                # Pad or truncate as needed
                precision = np.pad(precision, (0, max(0, num_classes - len(precision))))[
                    :num_classes
                ]
                recall = np.pad(recall, (0, max(0, num_classes - len(recall))))[:num_classes]
                f1 = np.pad(f1, (0, max(0, num_classes - len(f1))))[:num_classes]
                support = np.pad(support, (0, max(0, num_classes - len(support))))[:num_classes]

            # Create per-class metrics dictionary
            per_class_metrics = {}
            for i, class_name in enumerate(class_names):
                if i < len(precision):
                    # Calculate average confidence for this class
                    class_mask = targets == i
                    avg_confidence = (
                        float(np.mean(probabilities[class_mask, i])) if np.any(class_mask) else 0.0
                    )

                    per_class_metrics[class_name] = {
                        "precision": float(precision[i]),
                        "recall": float(recall[i]),
                        "f1_score": float(f1[i]),
                        "support": int(support[i]),
                        "avg_confidence": avg_confidence,
                    }
                else:
                    # Default values for missing classes
                    per_class_metrics[class_name] = {
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1_score": 0.0,
                        "support": 0,
                        "avg_confidence": 0.0,
                    }

            return per_class_metrics

        except Exception as e:
            logger.error(f"Error calculating per-class metrics: {e}")
            # Return default metrics for all classes
            return {
                class_name: {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0,
                    "support": 0,
                    "avg_confidence": 0.0,
                }
                for class_name in class_names
            }

    def evaluate_multitask(
        self, breed_classes: List[str], emotion_classes: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate a multitask model for both breed and emotion classification.

        Each task is evaluated completely separately with independent metrics,
        classification reports, and confusion matrices.

        Args:
            breed_classes: List of breed class names
            emotion_classes: List of emotion class names

        Returns:
            Dictionary containing evaluation metrics for both tasks
        """
        all_breed_targets = []
        all_breed_predictions = []
        all_breed_probabilities = []
        all_emotion_targets = []
        all_emotion_predictions = []
        all_emotion_probabilities = []
        total_time = 0
        num_samples = len(self.test_loader.dataset)

        logger.info(f"Starting multitask evaluation on {num_samples} samples")
        logger.info(f"Breed classes: {len(breed_classes)} classes")
        logger.info(f"Emotion classes: {len(emotion_classes)} classes")

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating multitask"):
                # Handle different batch formats
                if len(batch) == 3:
                    inputs, breed_targets, emotion_targets = batch
                elif len(batch) == 2:
                    # Fallback for single-task data - use same targets for both tasks
                    inputs, targets = batch
                    breed_targets = targets
                    emotion_targets = targets
                    logger.warning("Single-task data detected in multitask evaluation")
                else:
                    raise ValueError(f"Unexpected batch format with {len(batch)} elements")

                inputs = inputs.to(self.device)
                breed_targets = breed_targets.to(self.device)
                emotion_targets = emotion_targets.to(self.device)

                # Measure inference time
                start_time = time.time()
                try:
                    breed_logits = self.model(inputs, task="breed")
                    emotion_logits = self.model(inputs, task="emotion")
                except TypeError:
                    # Fallback if model doesn't support task argument
                    output = self.model(inputs)
                    if isinstance(output, tuple) and len(output) == 2:
                        breed_logits, emotion_logits = output
                    else:
                        raise ValueError(
                            "Model output format not supported for multitask evaluation"
                        )

                batch_time = time.time() - start_time
                total_time += batch_time

                # Get predictions and probabilities for breed
                breed_probabilities = torch.softmax(breed_logits, dim=1)
                _, breed_predictions = breed_logits.max(1)

                # Get predictions and probabilities for emotion
                emotion_probabilities = torch.softmax(emotion_logits, dim=1)
                _, emotion_predictions = emotion_logits.max(1)

                # Collect results
                all_breed_targets.extend(breed_targets.cpu().numpy())
                all_breed_predictions.extend(breed_predictions.cpu().numpy())
                all_breed_probabilities.extend(breed_probabilities.cpu().numpy())

                all_emotion_targets.extend(emotion_targets.cpu().numpy())
                all_emotion_predictions.extend(emotion_predictions.cpu().numpy())
                all_emotion_probabilities.extend(emotion_probabilities.cpu().numpy())

        # Convert to numpy arrays
        all_breed_targets = np.array(all_breed_targets)
        all_breed_predictions = np.array(all_breed_predictions)
        all_breed_probabilities = np.array(all_breed_probabilities)

        all_emotion_targets = np.array(all_emotion_targets)
        all_emotion_predictions = np.array(all_emotion_predictions)
        all_emotion_probabilities = np.array(all_emotion_probabilities)

        # Validate data integrity
        self._validate_multitask_data_integrity(
            all_breed_targets,
            all_breed_predictions,
            all_emotion_targets,
            all_emotion_predictions,
            breed_classes,
            emotion_classes,
        )

        # Calculate average inference time
        avg_inference_time = total_time / num_samples

        logger.info("=" * 60)
        logger.info("EVALUATING BREED CLASSIFICATION TASK")
        logger.info("=" * 60)

        # Calculate metrics for breed classification
        breed_results = self._calculate_task_metrics(
            all_breed_targets,
            all_breed_predictions,
            all_breed_probabilities,
            breed_classes,
            "breed",
            avg_inference_time,
        )

        logger.info("=" * 60)
        logger.info("EVALUATING EMOTION CLASSIFICATION TASK")
        logger.info("=" * 60)

        # Calculate metrics for emotion classification
        emotion_results = self._calculate_task_metrics(
            all_emotion_targets,
            all_emotion_predictions,
            all_emotion_probabilities,
            emotion_classes,
            "emotion",
            avg_inference_time,
        )

        # Validate data integrity
        self._validate_multitask_data_integrity(
            all_breed_targets,
            all_breed_predictions,
            all_emotion_targets,
            all_emotion_predictions,
            breed_classes,
            emotion_classes,
        )

        # Create combined results dictionary
        results = {
            "evaluation_type": "multitask",
            "num_samples": num_samples,
            "num_breed_classes": len(breed_classes),
            "num_emotion_classes": len(emotion_classes),
            "avg_inference_time_ms": float(avg_inference_time * 1000),
            "breed": breed_results,
            "emotion": emotion_results,
        }

        # Print comprehensive summary
        self._display_multitask_comparison(breed_results, emotion_results)

        logger.info("=" * 60)
        logger.info("MULTITASK EVALUATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total samples evaluated: {num_samples}")
        logger.info(f"Average inference time: {avg_inference_time * 1000:.2f} ms/sample")
        logger.info("")
        logger.info("BREED CLASSIFICATION RESULTS:")
        logger.info(f"  Accuracy: {breed_results['accuracy']:.4f}")
        logger.info(f"  Precision: {breed_results['precision']:.4f}")
        logger.info(f"  Recall: {breed_results['recall']:.4f}")
        logger.info(f"  F1-Score: {breed_results['f1_score']:.4f}")
        logger.info(f"  Top-3 Accuracy: {breed_results['top3_accuracy']:.4f}")
        logger.info(f"  Top-5 Accuracy: {breed_results['top5_accuracy']:.4f}")
        logger.info("")
        logger.info("EMOTION CLASSIFICATION RESULTS:")
        logger.info(f"  Accuracy: {emotion_results['accuracy']:.4f}")
        logger.info(f"  Precision: {emotion_results['precision']:.4f}")
        logger.info(f"  Recall: {emotion_results['recall']:.4f}")
        logger.info(f"  F1-Score: {emotion_results['f1_score']:.4f}")
        logger.info(f"  Top-3 Accuracy: {emotion_results['top3_accuracy']:.4f}")
        logger.info(f"  Top-5 Accuracy: {emotion_results['top5_accuracy']:.4f}")

        # Save results if output directory is provided
        if self.output_dir:
            # Save combined results
            results_path = self.output_dir / "multitask_evaluation_results.json"
            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)

            # Save individual task summaries
            breed_summary_path = self.output_dir / "breed_task_summary.json"
            with open(breed_summary_path, "w") as f:
                json.dump(breed_results, f, indent=4)

            emotion_summary_path = self.output_dir / "emotion_task_summary.json"
            with open(emotion_summary_path, "w") as f:
                json.dump(emotion_results, f, indent=4)

            logger.info(f"Multitask evaluation results saved to {self.output_dir}")
            logger.info(f"  - Combined results: {results_path}")
            logger.info(f"  - Breed task summary: {breed_summary_path}")
            logger.info(f"  - Emotion task summary: {emotion_summary_path}")

        return results

    def _calculate_task_metrics(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        class_names: List[str],
        task_name: str,
        avg_inference_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics for a single task.

        Args:
            targets: True labels
            predictions: Predicted labels
            probabilities: Prediction probabilities
            class_names: List of class names
            task_name: Name of the task (for file naming)
            avg_inference_time: Average inference time per sample

        Returns:
            Dictionary containing task metrics
        """
        try:
            # Calculate basic metrics
            accuracy = accuracy_score(targets, predictions)
            precision, recall, f1, support = precision_recall_fscore_support(
                targets, predictions, average="weighted", zero_division=0
            )

            # Top-k accuracy (only if we have enough classes)
            top3_accuracy = self._calculate_topk_accuracy(
                probabilities, targets, k=min(3, len(class_names))
            )
            top5_accuracy = self._calculate_topk_accuracy(
                probabilities, targets, k=min(5, len(class_names))
            )

            # Per-class metrics
            per_class_metrics = self._calculate_per_class_metrics(
                targets, predictions, probabilities, class_names
            )

            # Create confusion matrix
            cm_fig = plot_confusion_matrix(targets, predictions, class_names)

            # Print detailed classification report
            self._display_detailed_classification_report(
                targets, predictions, class_names, task_name
            )

            # Print classification report (for saving)
            class_report = print_classification_report(targets, predictions, class_names)

            # Save task-specific results if output directory is provided
            if self.output_dir:
                try:
                    # Save classification report
                    class_report_df = pd.DataFrame(class_report).transpose()
                    class_report_df.to_csv(
                        self.output_dir / f"{task_name}_classification_report.csv"
                    )

                    # Save per-class metrics
                    per_class_metrics_df = pd.DataFrame(per_class_metrics).transpose()
                    per_class_metrics_df.to_csv(
                        self.output_dir / f"{task_name}_per_class_metrics.csv"
                    )

                    # Save confusion matrix plot
                    cm_path = self.output_dir / f"{task_name}_confusion_matrix.png"
                    cm_fig.savefig(cm_path, dpi=300, bbox_inches="tight")
                    plt.close(cm_fig)  # Free memory

                except Exception as e:
                    logger.warning(f"Failed to save results for {task_name}: {e}")

            # Create results dictionary
            results = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "top3_accuracy": float(top3_accuracy),
                "top5_accuracy": float(top5_accuracy),
                "per_class_metrics": per_class_metrics,
                "class_report": class_report,
                "num_samples": len(targets),
                "num_classes": len(class_names),
            }

            # Add inference time if provided
            if avg_inference_time is not None:
                results["avg_inference_time_ms"] = float(avg_inference_time * 1000)

            # Save files with more descriptive logging
            if self.output_dir:
                logger.info(
                    f"  Saved {task_name} classification report to: "
                    f"{self.output_dir / f'{task_name}_classification_report.csv'}"
                )
                logger.info(
                    f"  Saved {task_name} per-class metrics to: "
                    f"{self.output_dir / f'{task_name}_per_class_metrics.csv'}"
                )
                logger.info(
                    f"  Saved {task_name} confusion matrix to: "
                    f"{self.output_dir / f'{task_name}_confusion_matrix.png'}"
                )

            return results

        except Exception as e:
            logger.error(f"Error calculating metrics for {task_name}: {e}")
            # Return minimal results in case of error
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "top3_accuracy": 0.0,
                "top5_accuracy": 0.0,
                "error": str(e),
            }

    def _display_detailed_classification_report(
        self, targets: np.ndarray, predictions: np.ndarray, class_names: List[str], task_name: str
    ) -> None:
        """
        Display a detailed classification report for a specific task.

        Args:
            targets: True labels
            predictions: Predicted labels
            class_names: List of class names
            task_name: Name of the task for display
        """
        from sklearn.metrics import classification_report

        logger.info(f"\n{task_name.upper()} CLASSIFICATION REPORT:")
        logger.info("=" * 80)

        # Generate detailed classification report
        report = classification_report(
            targets, predictions, target_names=class_names, output_dict=False, zero_division=0
        )

        # Log the report line by line for better formatting
        for line in report.split("\n"):
            if line.strip():  # Skip empty lines
                logger.info(line)

        logger.info("=" * 80)

    def evaluate_single_task(self, class_names: List[str]) -> Dict[str, Any]:
        """
        Backward compatibility method for single-task evaluation.

        This method maintains the original API while using the improved evaluation logic.

        Args:
            class_names: List of class names

        Returns:
            Dictionary containing evaluation metrics
        """
        return self.evaluate(class_names, task="breed")

    def auto_evaluate(
        self,
        breed_classes: Optional[List[str]] = None,
        emotion_classes: Optional[List[str]] = None,
        task: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Automatically determine evaluation type based on available class lists.

        Args:
            breed_classes: List of breed class names (optional)
            emotion_classes: List of emotion class names (optional)
            task: Specific task to evaluate ("breed", "emotion", or None for auto-detect)

        Returns:
            Dictionary containing evaluation metrics
        """
        # Auto-detect evaluation type
        if breed_classes and emotion_classes:
            # Both class lists provided - assume multitask
            logger.info(
                "Auto-detected multitask evaluation (both breed and emotion classes provided)"
            )
            return self.evaluate_multitask(breed_classes, emotion_classes)
        elif breed_classes and not emotion_classes:
            # Only breed classes - single task breed evaluation
            logger.info("Auto-detected single-task breed evaluation")
            return self.evaluate(breed_classes, task="breed")
        elif emotion_classes and not breed_classes:
            # Only emotion classes - single task emotion evaluation
            logger.info("Auto-detected single-task emotion evaluation")
            return self.evaluate(emotion_classes, task="emotion")
        elif task and (breed_classes or emotion_classes):
            # Specific task requested
            class_names = breed_classes if task == "breed" else emotion_classes
            if class_names:
                logger.info(f"Evaluating specific task: {task}")
                return self.evaluate(class_names, task=task)
            else:
                raise ValueError(f"Class names for task '{task}' not provided")
        else:
            raise ValueError("Must provide at least one of breed_classes or emotion_classes")

    def _display_multitask_comparison(
        self, breed_results: Dict[str, Any], emotion_results: Dict[str, Any]
    ) -> None:
        """
        Display a side-by-side comparison of breed and emotion task results.

        Args:
            breed_results: Results from breed classification
            emotion_results: Results from emotion classification
        """
        logger.info("\nMULTITASK PERFORMANCE COMPARISON:")
        logger.info("=" * 80)
        logger.info(f"{'Metric':<20} {'Breed Task':<15} {'Emotion Task':<15} {'Difference':<15}")
        logger.info("-" * 80)

        metrics_to_compare = [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "top3_accuracy",
            "top5_accuracy",
        ]

        for metric in metrics_to_compare:
            if metric in breed_results and metric in emotion_results:
                breed_val = breed_results[metric]
                emotion_val = emotion_results[metric]
                diff = breed_val - emotion_val
                diff_str = f"{diff:+.4f}"

                logger.info(
                    f"{metric.replace('_', ' ').title():<20} {breed_val:<15.4f} "
                    f"{emotion_val:<15.4f} {diff_str:<15}"
                )

        logger.info("=" * 80)

        # Determine which task performed better
        breed_acc = breed_results.get("accuracy", 0)
        emotion_acc = emotion_results.get("accuracy", 0)

        if breed_acc > emotion_acc:
            logger.info(
                f"🏆 Breed classification performed better (Δ accuracy: +{breed_acc - emotion_acc:.4f})"
            )
        elif emotion_acc > breed_acc:
            logger.info(
                f"🏆 Emotion classification performed better (Δ accuracy: +{emotion_acc - breed_acc:.4f})"
            )
        else:
            logger.info("🤝 Both tasks achieved equal accuracy")

    def _validate_multitask_data_integrity(
        self,
        breed_targets: np.ndarray,
        breed_predictions: np.ndarray,
        emotion_targets: np.ndarray,
        emotion_predictions: np.ndarray,
        breed_classes: List[str],
        emotion_classes: List[str],
    ) -> None:
        """
        Validate that multitask evaluation data has proper integrity.

        Args:
            breed_targets: Breed true labels
            breed_predictions: Breed predictions
            emotion_targets: Emotion true labels
            emotion_predictions: Emotion predictions
            breed_classes: Breed class names
            emotion_classes: Emotion class names
        """
        logger.info("Validating multitask evaluation data integrity...")

        # Check array lengths
        assert len(breed_targets) == len(
            breed_predictions
        ), "Breed targets/predictions length mismatch"
        assert len(emotion_targets) == len(
            emotion_predictions
        ), "Emotion targets/predictions length mismatch"
        assert len(breed_targets) == len(emotion_targets), "Sample count mismatch between tasks"

        # Check class ranges
        assert breed_targets.min() >= 0 and breed_targets.max() < len(
            breed_classes
        ), f"Breed targets out of range [0, {len(breed_classes)-1}]"
        assert emotion_targets.min() >= 0 and emotion_targets.max() < len(
            emotion_classes
        ), f"Emotion targets out of range [0, {len(emotion_classes)-1}]"
        assert breed_predictions.min() >= 0 and breed_predictions.max() < len(
            breed_classes
        ), f"Breed predictions out of range [0, {len(breed_classes)-1}]"
        assert emotion_predictions.min() >= 0 and emotion_predictions.max() < len(
            emotion_classes
        ), f"Emotion predictions out of range [0, {len(emotion_classes)-1}]"

        # Log validation success
        logger.info(f"✓ Data integrity validated:")
        logger.info(f"  - {len(breed_targets)} samples for both tasks")
        logger.info(f"  - Breed classes: {len(breed_classes)} (range: 0-{len(breed_classes)-1})")
        logger.info(
            f"  - Emotion classes: {len(emotion_classes)} (range: 0-{len(emotion_classes)-1})"
        )
        logger.info(
            f"  - Breed target distribution: {np.bincount(breed_targets, minlength=len(breed_classes))}"
        )
        logger.info(
            f"  - Emotion target distribution: {np.bincount(emotion_targets, minlength=len(emotion_classes))}"
        )


def create_evaluator_from_config(
    model: nn.Module,
    test_loader: DataLoader,
    config: Dict[str, Any],
    device: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Evaluator:
    """
    Factory function to create an Evaluator from configuration.

    Args:
        model: Model to evaluate
        test_loader: Test data loader
        config: Configuration dictionary
        device: Device to use (overrides config if provided)
        output_dir: Output directory (overrides config if provided)

    Returns:
        Configured Evaluator instance
    """
    # Get device from config or parameter
    eval_device = device or config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # Get output directory from config or parameter
    eval_output_dir = output_dir or config.get("output_dir")

    return Evaluator(
        model=model,
        test_loader=test_loader,
        device=eval_device,
        output_dir=eval_output_dir,
    )


def evaluate_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    test_loader: DataLoader,
    class_names: List[str],
    model_config: Dict[str, Any],
    task: str = "breed",
    device: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Convenience function to evaluate a model directly from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        test_loader: Test data loader
        class_names: List of class names
        model_config: Model configuration
        task: Task to evaluate
        device: Device to use
        output_dir: Directory to save results

    Returns:
        Dictionary containing evaluation metrics
    """
    from engine.model import load_model

    # Load model from checkpoint
    model = load_model(
        path=str(checkpoint_path),
        num_classes=len(class_names),
        model_type=task,
        model_config=model_config,
    )

    # Create evaluator
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        output_dir=output_dir,
    )

    # Run evaluation
    return evaluator.evaluate(class_names, task=task)
