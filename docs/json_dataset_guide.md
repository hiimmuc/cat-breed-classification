# JSON Dataset Training Guide

This guide explains how to use JSON dataset files for training cat breed and emotion classification models.

## Overview

The JSON dataset feature allows you to train models using pre-prepared JSON files that contain:

-   Image paths and labels for both breed and emotion classification
-   Metadata about classes and dataset splits
-   Confidence scores and additional information

This is particularly useful when you have already processed and labeled your data, or when working with datasets that combine multiple tasks.

## JSON Dataset Format

The expected JSON format is:

```json
{
  "metadata": {
    "split": "train",
    "total_samples": 7734,
    "breed_classes": ["Abyssinian", "American Bobtail", ...],
    "emotion_classes": ["Angry", "Other", "Sad", "happy"],
    "num_breed_classes": 21,
    "num_emotion_classes": 4,
    "created_at": "2025-06-27 00:48:28"
  },
  "samples": [
    {
      "image_path": "Russian Blue/Russian Blue_24.jpg",
      "breed_label": 15,
      "breed_class": "Russian Blue",
      "emotion_label": 0,
      "emotion_class": "Angry",
      "emotion_confidence": 0.748
    },
    ...
  ]
}
```

## Quick Start

### 1. Multitask Training with JSON

```bash
python src/main.py train \
    --mode multitask \
    --use-json-dataset \
    --train-json data/json_datasets_with_emotions/train_dataset_with_emotions.json \
    --val-json data/json_datasets_with_emotions/val_dataset_with_emotions.json \
    --test-json data/json_datasets_with_emotions/test_dataset_with_emotions.json \
    --json-data-root data/data-breed/processed \
    --epochs 50 \
    --batch-size 32
```

### 2. Using JSON Configuration File

```bash
python src/main.py train --config-path config/json_multitask_config.yaml
```

### 3. Single Task Training with JSON

For breed classification only:

```bash
python src/main.py train \
    --mode breed \
    --use-json-dataset \
    --train-json data/json_datasets_with_emotions/train_dataset_with_emotions.json \
    --val-json data/json_datasets_with_emotions/val_dataset_with_emotions.json \
    --test-json data/json_datasets_with_emotions/test_dataset_with_emotions.json \
    --json-data-root data/data-breed/processed
```

For emotion classification only:

```bash
python src/main.py train \
    --mode emotion \
    --use-json-dataset \
    --train-json data/json_datasets_with_emotions/train_dataset_with_emotions.json \
    --val-json data/json_datasets_with_emotions/val_dataset_with_emotions.json \
    --test-json data/json_datasets_with_emotions/test_dataset_with_emotions.json \
    --json-data-root data/data-breed/processed
```

## Configuration Options

### Command Line Arguments

#### Required for JSON Dataset:

-   `--use-json-dataset`: Enable JSON dataset mode
-   `--train-json`: Path to training JSON file
-   `--val-json`: Path to validation JSON file
-   `--test-json`: Path to test JSON file
-   `--json-data-root`: Root directory containing the images

#### Optional Arguments:

-   All standard training arguments (epochs, batch-size, lr, etc.)
-   Multitask-specific arguments (breed-weight, emotion-weight, shared-features)

### Configuration File Format

```yaml
# JSON Dataset Configuration
mode: "multitask"

# JSON dataset settings
use_json_dataset: true
train_json: "data/json_datasets_with_emotions/train_dataset_with_emotions.json"
val_json: "data/json_datasets_with_emotions/val_dataset_with_emotions.json"
test_json: "data/json_datasets_with_emotions/test_dataset_with_emotions.json"
json_data_root: "data/data-breed/processed"

# Model and training settings
backbone: "mobilenet_v2"
epochs: 50
batch_size: 32
learning_rate: 0.001
shared_features: true
breed_weight: 1.0
emotion_weight: 1.0
```

## Advantages of JSON Datasets

### 1. **Pre-processed Labels**

-   Both breed and emotion labels are already prepared
-   No need to maintain separate directory structures
-   Confidence scores and metadata available

### 2. **Flexible Data Organization**

-   Images can be organized in any directory structure
-   Relative paths in JSON make datasets portable
-   Easy to subset or modify datasets

### 3. **Consistent Splits**

-   Train/val/test splits are pre-defined
-   Reproducible experiments across runs
-   No random splitting variability

### 4. **Rich Metadata**

-   Class mappings stored with dataset
-   Creation timestamps and versioning
-   Additional annotations can be included

## Data Preparation

### Creating JSON Datasets

If you need to create JSON datasets from directories, you can use the existing data utilities:

```python
from src.utils.data_utils import create_json_datasets

# This function would need to be implemented to convert
# directory-based datasets to JSON format
create_json_datasets(
    breed_data_dir="data/data-breed/processed",
    emotion_data_dir="data/data-emo/processed",
    output_dir="data/json_datasets_with_emotions",
    emotion_model_path="path/to/emotion/model.pth"
)
```

### File Organization

Recommended directory structure:

```
project/
├── data/
│   ├── data-breed/processed/          # Original breed images
│   └── json_datasets_with_emotions/   # JSON dataset files
│       ├── train_dataset_with_emotions.json
│       ├── val_dataset_with_emotions.json
│       └── test_dataset_with_emotions.json
├── config/
│   └── json_multitask_config.yaml     # JSON config
└── src/
    └── main.py                        # Training script
```

## Example Workflows

### Complete Training Pipeline

```bash
# 1. Train multitask model with JSON dataset
python src/main.py train --config-path config/json_multitask_config.yaml

# 2. Evaluate the model (will automatically use JSON dataset if configured)
python src/main.py evaluate --config-path config/json_multitask_config.yaml

# 3. Make predictions
python src/main.py predict \
    --model-path src/checkpoints/multitask_*/best_state.pth \
    --input tests/resources/test.jpg
```

### Custom Training Configuration

```bash
# Train with custom settings
python src/main.py train \
    --mode multitask \
    --use-json-dataset \
    --train-json data/json_datasets_with_emotions/train_dataset_with_emotions.json \
    --val-json data/json_datasets_with_emotions/val_dataset_with_emotions.json \
    --test-json data/json_datasets_with_emotions/test_dataset_with_emotions.json \
    --json-data-root data/data-breed/processed \
    --backbone efficientnet_b0 \
    --epochs 100 \
    --batch-size 64 \
    --lr 0.0005 \
    --breed-weight 1.2 \
    --emotion-weight 0.8 \
    --no-shared-features
```

## Performance Considerations

### Memory Usage

-   JSON datasets load metadata into memory but images on-demand
-   Similar memory footprint to directory-based datasets
-   Consider image caching for very large datasets

### Loading Speed

-   Initial JSON parsing is fast
-   Image loading speed depends on storage and preprocessing
-   Use appropriate num_workers for your system

### Dataset Size

-   JSON files scale well to large datasets
-   Consider splitting very large datasets across multiple JSON files
-   Metadata overhead is minimal compared to image data

## Troubleshooting

### Common Issues

1. **File Not Found Errors**:

    - Check that all JSON files exist
    - Verify json_data_root points to correct directory
    - Ensure image paths in JSON are relative to data root

2. **Image Loading Errors**:

    - Missing image files referenced in JSON
    - Corrupted image files
    - Incorrect path separators (use forward slashes)

3. **Memory Issues**:

    - Reduce batch size
    - Decrease num_workers
    - Use smaller image size

4. **Label Mismatches**:
    - Verify class indices match class names
    - Check that label ranges are valid
    - Ensure consistent class mappings across splits

### Validation

Test your JSON dataset before training:

```bash
python test_json_dataset.py
```

This will validate:

-   JSON file loading
-   Image accessibility
-   Data loader creation
-   Configuration compatibility

## Integration with Existing Workflows

### Compatibility

-   ✅ **Training**: Full support for all modes (breed, emotion, multitask)
-   ✅ **Configuration**: YAML config file support
-   ✅ **Logging**: TensorBoard and console logging
-   ✅ **Checkpointing**: Standard model saving and loading
-   ⚠️ **Evaluation**: May need updates for JSON dataset paths
-   ⚠️ **Prediction**: Works with saved models, independent of dataset format

### Migration from Directory-based Training

To switch from directory-based to JSON-based training:

1. **Create JSON datasets** from your existing directories
2. **Update configuration** to use JSON settings
3. **Verify paths** are correct for your setup
4. **Test with small dataset** before full training

The JSON dataset feature provides a flexible and efficient way to train models with pre-processed, multi-task datasets while maintaining full compatibility with existing training pipelines.
