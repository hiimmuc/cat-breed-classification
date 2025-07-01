"""Model definitions for cat breed classification."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary
from torchview import draw_graph

logger = logging.getLogger(__name__)

LIST_BACKBONES = models.list_models()


def get_backbone_features(backbone: nn.Module) -> int:
    """Extract feature dimension from various backbone architectures."""
    if hasattr(backbone, "fc"):
        return backbone.fc.in_features
    elif hasattr(backbone, "classifier") and hasattr(backbone.classifier, "in_features"):
        return backbone.classifier.in_features
    elif hasattr(backbone, "classifier") and isinstance(backbone.classifier, nn.Sequential):
        return backbone.classifier[-1].in_features
    elif hasattr(backbone, "head"):
        return backbone.head.in_features
    elif hasattr(backbone, "heads"):
        return backbone.heads[-1].in_features
    else:
        raise ValueError(f"Unsupported backbone architecture: {type(backbone)}")


def replace_classifier_head(backbone: nn.Module) -> None:
    """Replace the final classification layer with identity."""
    if hasattr(backbone, "fc"):
        backbone.fc = nn.Identity()
    elif hasattr(backbone, "classifier"):
        if isinstance(backbone.classifier, nn.Sequential):
            backbone.classifier[-1] = nn.Identity()
        else:
            backbone.classifier = nn.Identity()
    elif hasattr(backbone, "head"):
        backbone.head = nn.Identity()
    elif hasattr(backbone, "heads"):
        backbone.heads[-1] = nn.Identity()


class BaseClassifier(nn.Module):
    """Base classifier with common functionality."""

    def __init__(self, num_classes: int, backbone: str = "mobilenet_v2", pretrained: bool = True):
        super().__init__()
        self._backbone_name = backbone

        if backbone not in LIST_BACKBONES:
            raise ValueError(
                f"Unsupported backbone: {backbone}. Available: {LIST_BACKBONES[:10]}..."
            )

        # Load backbone
        weights = "DEFAULT" if pretrained else None
        self.backbone = models.get_model(backbone, weights=weights)
        summary(self.backbone, (3, 224, 224), device="cpu", depth=1)

        # Get feature dimension and replace head
        feature_dim = get_backbone_features(self.backbone)
        replace_classifier_head(self.backbone)

        # Dropout layer for regularization
        # self.dropout = nn.Dropout(0.2)

        # Create new classifier
        self.classifier = nn.Linear(feature_dim, num_classes)

        logger.info(f"Created {backbone} model with {num_classes} classes")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone and classifier."""
        features = self.backbone(x)
        # features = self.dropout(features)
        return self.classifier(features)


class CatBreedClassifier(BaseClassifier):
    """Cat breed classification model."""

    __type__ = "breed"


class CatEmotionClassifier(BaseClassifier):
    """Cat emotion classification model."""

    __type__ = "emotion"


class CatMultiTaskClassifier(BaseClassifier):
    """Multitask classifier for both breed and emotion classification."""

    __type__ = "multitask"

    def __init__(
        self,
        num_breed_classes: int,
        num_emotion_classes: int,
        backbone: str = "mobilenet_v2",
        pretrained: bool = False,
        shared_features: bool = False,
    ):
        """
        Initialize the multitask model.

        Args:
            num_breed_classes: Number of breed classes
            num_emotion_classes: Number of emotion classes
            backbone: Model backbone (e.g. mobilenet_v2, efficientnet_b0)
            pretrained: Whether to use pre-trained weights
            shared_features: Whether to share features between tasks
        """
        # Initialize without calling parent __init__ to avoid duplicate processing
        nn.Module.__init__(self)
        self._backbone_name = backbone
        self.shared_features = shared_features

        if backbone not in LIST_BACKBONES:
            raise ValueError(
                f"Unsupported backbone: {backbone}. Available: {LIST_BACKBONES[:10]}..."
            )

        # Load backbone
        weights = "DEFAULT" if pretrained else None
        self.backbone = models.get_model(backbone, weights=weights)

        # Get feature dimension before replacing head
        feature_dim = get_backbone_features(self.backbone)
        replace_classifier_head(self.backbone)

        if shared_features:
            # Shared backbone with separate heads
            self.breed_head = nn.Sequential(
                nn.LeakyReLU(), nn.Dropout(0.1), nn.Linear(feature_dim, num_breed_classes)
            )
            self.emotion_head = nn.Sequential(
                nn.LeakyReLU(), nn.Dropout(0.1), nn.Linear(feature_dim, num_emotion_classes)
            )
        else:
            # Task-specific feature extractors
            hidden_dim = feature_dim // 2
            self.breed_feature_extractor = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(0.1),
            )
            self.emotion_feature_extractor = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(0.1),
            )
            self.breed_head = nn.Linear(hidden_dim, num_breed_classes)
            self.emotion_head = nn.Linear(hidden_dim, num_emotion_classes)

        logger.info(
            f"Created multitask {backbone}: {num_breed_classes} breeds, {num_emotion_classes} emotions (shared={shared_features})"
        )

    def forward(
        self, x: torch.Tensor, task: str = "breed"
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with task selection.

        Args:
            x: Input tensor
            task: Which task to compute ("breed", "emotion")

        Returns:
            Task logits (breed or emotion)
        """
        features = self.backbone(x)

        if self.shared_features:
            breed_logits = self.breed_head(features)
            emotion_logits = self.emotion_head(features)
        else:
            breed_logits = self.breed_head(self.breed_feature_extractor(features))
            emotion_logits = self.emotion_head(self.emotion_feature_extractor(features))

        return breed_logits if task == "breed" else emotion_logits

    def get_task_parameters(self, task: str) -> List[torch.nn.Parameter]:
        """
        Get parameters for a specific task (useful for task-specific optimization).

        Args:
            task: Task name ("breed" or "emotion")

        Returns:
            List of parameters for the specified task
        """
        if task == "breed":
            if self.shared_features:
                return list(self.breed_head.parameters())
            else:
                return list(self.breed_feature_extractor.parameters()) + list(
                    self.breed_head.parameters()
                )

        elif task == "emotion":
            if self.shared_features:
                return list(self.emotion_head.parameters())
            else:
                return list(self.emotion_feature_extractor.parameters()) + list(
                    self.emotion_head.parameters()
                )

        else:
            raise ValueError(f"Invalid task: {task}. Must be 'breed' or 'emotion'")

    def get_shared_parameters(self) -> List[torch.nn.Parameter]:
        """Get shared backbone parameters."""
        return list(self.backbone.parameters())


def create_model(
    num_classes: int,
    model_config: Optional[Dict] = None,
    model_type: str = "breed",
    num_emotion_classes: Optional[int] = None,
) -> nn.Module:
    """Create a cat classifier model with simplified configuration."""
    config = model_config or {}

    model_classes = {
        "breed": CatBreedClassifier,
        "emotion": CatEmotionClassifier,
        "multitask": CatMultiTaskClassifier,
    }

    if model_type not in model_classes:
        raise ValueError(
            f"Invalid model_type: {model_type}. Choose from {list(model_classes.keys())}"
        )

    if model_type == "multitask":
        if num_emotion_classes is None:
            raise ValueError("num_emotion_classes is required for multitask model")
        return model_classes[model_type](
            num_breed_classes=num_classes,
            num_emotion_classes=num_emotion_classes,
            **{
                k: v
                for k, v in config.items()
                if k in ["backbone", "pretrained", "shared_features"]
            },
        )
    else:
        return model_classes[model_type](
            num_classes=num_classes,
            **{k: v for k, v in config.items() if k in ["backbone", "pretrained"]},
        )


def load_model(
    path: Union[str, Path],
    num_classes: int,
    model_config: Optional[Dict] = None,
    model_type: str = "breed",
    num_emotion_classes: Optional[int] = None,
) -> nn.Module:
    """Load model from checkpoint with automatic handling."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Load checkpoint data
    checkpoint = torch.load(path, map_location="cpu")

    # Create model architecture
    model = create_model(
        num_classes=num_classes,
        model_config=model_config,
        model_type=model_type,
        num_emotion_classes=num_emotion_classes,
    )

    # Load weights
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(
            f"Loaded {model_type} model from checkpoint: epoch {checkpoint.get('epoch', 'unknown')}"
        )
    else:
        # Handle legacy format where entire model was saved
        model = checkpoint
        logger.info(f"Loaded complete {model_type} model from {os.path.relpath(path)}")

    return model


def save_model_summary(
    model: nn.Module, save_path: Union[str, Path], input_size: tuple = (3, 224, 224)
) -> None:
    """Save model summary and architecture visualization."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Create model graph
        model_graph = draw_graph(
            model, input_size=input_size, save_graph=True, filename=str(save_path)
        )
        logger.info(f"Model architecture saved to {os.path.relpath(save_path)}")
    except Exception as e:
        logger.warning(f"Could not save model visualization: {e}")

        # Fallback: save text summary
        try:
            with save_path.with_suffix(".txt").open("w") as f:
                f.write(str(model))
            logger.info(
                f"Model summary saved as text to {os.path.relpath(save_path.with_suffix('.txt'))}"
            )
        except Exception as e2:
            logger.error(f"Could not save model summary: {e2}")


if __name__ == "__main__":
    # Example usage and testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create and test breed model
    breed_model = create_model(num_classes=21, model_type="breed")
    print(f"Breed model: {breed_model}")

    # Create and test multitask model
    multitask_model = create_model(num_classes=21, num_emotion_classes=4, model_type="multitask")
    print(f"Multitask model: {multitask_model}")

    # Test forward pass
    sample_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        breed_output = breed_model(sample_input)
        multitask_output = multitask_model(sample_input)
        print(f"Breed output shape: {breed_output.shape}")
        print(f"Multitask output shapes: {[out.shape for out in multitask_output]}")
