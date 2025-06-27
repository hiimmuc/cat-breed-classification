# Multitask Training Guide

This guide explains how to use the multitask training mode for joint cat breed and emotion classification.

## Overview

The multitask mode allows you to train a single model that can simultaneously predict:

-   Cat breed (e.g., Persian, Siamese, Bengal, etc.)
-   Cat emotion (e.g., Happy, Sad, Angry, Other)

The model uses a shared backbone with two task-specific heads, allowing it to learn features that are useful for both tasks.

## Quick Start

### 1. Basic Multitask Training

```bash
python src/main.py train \
    --mode multitask \
    --breed-data-dir data/data-breed/processed \
    --emotion-data-dir data/data-emo/processed \
    --epochs 50 \
    --batch-size 32
```

### 2. Using Configuration File

Create or use the provided `config/multitask_config.yaml`:

```bash
python src/main.py train --config-path config/multitask_config.yaml
```

### 3. Training Individual Tasks

For breed classification only:

```bash
python src/main.py train --mode breed --data-dir data/data-breed/processed
```

For emotion classification only:

```bash
python src/main.py train --mode emotion --data-dir data/data-emo/processed
```

## Configuration Options

### Command Line Arguments

#### Required for Multitask:

-   `--mode multitask`: Enable multitask training
-   `--breed-data-dir`: Path to breed classification dataset
-   `--emotion-data-dir`: Path to emotion classification dataset

#### Optional Multitask Settings:

-   `--breed-weight 1.0`: Weight for breed classification loss
-   `--emotion-weight 1.0`: Weight for emotion classification loss
-   `--shared-features`: Use shared feature extractor (default: True)

#### Model Settings:

-   `--backbone mobilenet_v2`: Model architecture
-   `--epochs 50`: Number of training epochs
-   `--batch-size 32`: Batch size
-   `--lr 0.001`: Learning rate
-   `--weight-decay 0.0001`: Weight decay

### Configuration File Format

```yaml
# Mode: "breed", "emotion", or "multitask"
mode: "multitask"

# Data paths
breed_data_dir: "data/data-breed/processed"
emotion_data_dir: "data/data-emo/processed"

# Model configuration
backbone: "mobilenet_v2"
shared_features: true
breed_weight: 1.0
emotion_weight: 1.0

# Training parameters
epochs: 50
batch_size: 32
learning_rate: 0.001
weight_decay: 0.0001
```

## Model Architecture

### Multitask Model Features:

-   **Shared Backbone**: Pre-trained CNN (MobileNet, EfficientNet, etc.)
-   **Task-Specific Heads**: Separate classification layers for breed and emotion
-   **Flexible Architecture**: Can use shared or separate feature extractors

### Model Types:

1. **Shared Features** (default): Single feature extractor → Two heads
2. **Separate Features**: Two feature extractors → Two heads

## Training Process

### Data Loading:

-   Loads both breed and emotion datasets
-   Creates multitask data loaders that yield (image, breed_label, emotion_label)
-   Balances datasets using configurable strategies

### Loss Computation:

-   Computes separate losses for breed and emotion predictions
-   Combines losses with configurable weights
-   Supports different loss functions (CrossEntropy, Focal loss)

### Checkpointing:

-   Saves best model based on combined validation loss
-   Stores both breed and emotion class names
-   Compatible with existing evaluation tools

## Example Usage

### Complete Training Pipeline:

```bash
# 1. Train multitask model
python src/main.py train \
    --mode multitask \
    --breed-data-dir data/data-breed/processed \
    --emotion-data-dir data/data-emo/processed \
    --backbone mobilenet_v2 \
    --epochs 100 \
    --batch-size 64 \
    --lr 0.001 \
    --breed-weight 1.0 \
    --emotion-weight 1.5 \
    --use-tensorboard

# 2. Evaluate the model
python src/main.py evaluate \
    --mode multitask \
    --model-path src/checkpoints/multitask_mobilenet_v2_*/best_state.pth

# 3. Make predictions
python src/main.py predict \
    --mode multitask \
    --model-path src/checkpoints/multitask_mobilenet_v2_*/best_state.pth \
    --input tests/resources/test.jpg
```

## Advanced Configuration

### Loss Weighting Strategies:

```bash
# Equal weights
--breed-weight 1.0 --emotion-weight 1.0

# Emphasize emotion
--breed-weight 0.7 --emotion-weight 1.3

# Emphasize breed
--breed-weight 1.5 --emotion-weight 0.8
```

### Architecture Variants:

```bash
# Shared features (default)
--shared-features

# Separate feature extractors
--no-shared-features
```

### Different Backbones:

```bash
# Lightweight
--backbone mobilenet_v3_small

# Balanced
--backbone mobilenet_v2

# Larger capacity
--backbone efficientnet_b0
```

## Monitoring Training

### TensorBoard:

```bash
tensorboard --logdir src/checkpoints/multitask_*
```

Tracks:

-   Combined validation loss
-   Individual task losses
-   Task-specific accuracies
-   Learning rate schedules

### Console Output:

-   Real-time training progress
-   Loss breakdowns per task
-   Validation metrics
-   Best model updates

## Output Structure

```
src/checkpoints/multitask_mobilenet_v2_20250627_*/
├── best_state.pth              # Best model weights
├── last_state.pth              # Latest model weights
├── class_names.json            # Class mappings for both tasks
├── training_config.yaml        # Training configuration
├── learning_curves.png         # Training plots
└── evaluation/                 # Evaluation results
    ├── breed_confusion_matrix.png
    ├── emotion_confusion_matrix.png
    └── metrics.json
```

## Performance Tips

### Data Balance:

-   Ensure both datasets have sufficient samples
-   Consider dataset size ratio when setting loss weights
-   Use data augmentation for smaller datasets

### Memory Optimization:

-   Reduce batch size if GPU memory is limited
-   Use mixed precision training for larger models
-   Consider gradient accumulation for small batch sizes

### Hyperparameter Tuning:

-   Start with equal loss weights (1.0, 1.0)
-   Adjust based on relative task importance
-   Monitor individual task performances

## Troubleshooting

### Common Issues:

1. **Memory Errors**:

    - Reduce batch size
    - Use smaller backbone model
    - Enable gradient checkpointing

2. **Unbalanced Training**:

    - Adjust loss weights
    - Check dataset sizes
    - Monitor individual task losses

3. **Poor Convergence**:
    - Lower learning rate
    - Increase patience for scheduler
    - Check data quality

### Model Loading:

-   Ensure model path includes all required components
-   Check compatibility with evaluation scripts
-   Verify class name mappings

## Integration

The multitask mode is fully integrated with existing tools:

-   ✅ Training pipeline (main.py)
-   ✅ Model architectures (model.py)
-   ✅ Configuration system (YAML configs)
-   ✅ Checkpointing and resuming
-   ✅ TensorBoard logging
-   ⚠️ Evaluation (single-task compatible)
-   ⚠️ Prediction (requires updates)

Future updates will extend evaluation and prediction tools to fully support multitask models.
