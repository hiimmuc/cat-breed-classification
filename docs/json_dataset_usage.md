# JSON Dataset Management

This document describes the enhanced dataset management system that uses JSON files instead of loading directly from directories.

## Overview

The new system provides better control, reproducibility, and performance for dataset management by:

-   Creating JSON files with dataset splits and metadata
-   Loading data from JSON files instead of scanning directories
-   Ensuring reproducible dataset splits across runs
-   Supporting metadata preservation and versioning

## Key Features

### ✨ Benefits

-   **Reproducible Splits**: Same train/val/test splits every time
-   **Faster Loading**: No directory scanning needed
-   **Metadata Preservation**: Stores creation date, splits, class info
-   **Version Control**: Easy to track dataset versions
-   **Multi-format Support**: JPG, JPEG, PNG, BMP, TIFF
-   **Error Handling**: Better error reporting and fallbacks
-   **Portability**: Relative paths make datasets portable

### 📁 File Structure

When you create JSON datasets, you'll get these files:

```
json_datasets/
├── train_dataset.json      # Training data (70% by default)
├── val_dataset.json        # Validation data (15% by default)
├── test_dataset.json       # Test data (15% by default)
└── total_dataset.json      # All data combined
```

### 📋 JSON Format

Each JSON file contains:

```json
{
  "samples": [
    {
      "image_path": "Abyssinian/image_001.jpg",
      "label": 0,
      "class_name": "Abyssinian"
    }
  ],
  "split": "train",
  "size": 1000,
  "metadata": {
    "class_names": ["Abyssinian", "Bengal", ...],
    "class_to_idx": {"Abyssinian": 0, "Bengal": 1, ...},
    "num_classes": 21,
    "data_dir": "/path/to/data",
    "created_at": "2025-06-26 10:30:00",
    "random_seed": 42,
    "val_ratio": 0.15,
    "test_ratio": 0.15
  }
}
```

## Usage Guide

### 1. Creating JSON Datasets

#### Using the Script

```bash
# Create JSON datasets from directory structure
python src/create_dataset_json.py \
    --data_dir data/data-breed/processed \
    --output_dir data/json_datasets \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --random_seed 42
```

#### Using the Function

```python
from utils.data_utils import create_dataset_json

stats = create_dataset_json(
    data_dir="data/data-breed/processed",
    output_dir="data/json_datasets",
    val_ratio=0.15,
    test_ratio=0.15,
    random_seed=42
)
```

### 2. Loading Data from JSON

#### Full Dataset Loading

```python
from utils.data_utils import get_data_loaders

# Load all splits from JSON files
data_loaders = get_data_loaders(
    json_dir="data/json_datasets",
    batch_size=32,
    num_workers=4
)

train_loader = data_loaders["train"]
val_loader = data_loaders["val"]
test_loader = data_loaders["test"]
class_names = data_loaders["class_names"]
```

#### Test-Only Loading

```python
from utils.data_utils import get_test_loader_from_json

# Load only test data
test_info = get_test_loader_from_json(
    json_path="data/json_datasets/test_dataset.json",
    batch_size=32
)

test_loader = test_info["test_loader"]
class_names = test_info["class_names"]
```

### 3. Running Tests from JSON

#### Using the Script

```bash
# Run test evaluation from JSON
python src/test_from_json.py \
    --json_path data/json_datasets/test_dataset.json \
    --model_path checkpoints/best_model.pth \
    --batch_size 32
```

#### Demo Mode (without model)

```bash
# Just show data loading capabilities
python src/test_from_json.py \
    --json_path data/json_datasets/test_dataset.json
```

### 4. Dataset Analysis

```python
from utils.data_utils import analyze_dataset_json, print_dataset_analysis

# Analyze dataset
analysis = analyze_dataset_json("data/json_datasets/train_dataset.json")
print_dataset_analysis(analysis)
```

## Migration Guide

### From Directory Loading

**Old way:**

```python
data_loaders = get_data_loaders(
    data_dir="data/processed",
    batch_size=32
)
```

**New way:**

```python
# First, create JSON files (one time)
create_dataset_json(
    data_dir="data/processed",
    output_dir="data/json_datasets"
)

# Then load from JSON
data_loaders = get_data_loaders(
    json_dir="data/json_datasets",
    batch_size=32
)
```

### Backward Compatibility

The old directory-based loading still works:

```python
# Legacy method still supported
data_loaders = get_data_loaders(
    data_dir="data/processed",
    batch_size=32
)
```

## API Reference

### Functions

#### `create_dataset_json()`

Creates JSON dataset files from directory structure.

**Parameters:**

-   `data_dir`: Directory with class folders
-   `output_dir`: Where to save JSON files
-   `val_ratio`: Validation split ratio (default: 0.1)
-   `test_ratio`: Test split ratio (default: 0.1)
-   `random_seed`: For reproducibility (default: 42)

#### `get_data_loaders()`

Creates data loaders from JSON or directory.

**Parameters:**

-   `json_dir`: Directory with JSON files (preferred)
-   `data_dir`: Directory with class folders (legacy)
-   `batch_size`: Batch size (default: 32)
-   `num_workers`: Data loading workers (default: 4)
-   `img_size`: Image size (default: 224)

#### `get_test_loader_from_json()`

Creates test loader from JSON file.

**Parameters:**

-   `json_path`: Path to JSON dataset file
-   `batch_size`: Batch size (default: 32)
-   `num_workers`: Data loading workers (default: 4)
-   `img_size`: Image size (default: 224)

#### `CatBreedDataset`

Dataset class supporting both JSON and directory loading.

**Parameters:**

-   `json_path`: Path to JSON file (preferred)
-   `data_dir`: Directory with classes (legacy)
-   `transform`: Image transforms
-   Additional parameters for legacy mode

## Examples

See the example scripts:

-   `src/example_json_usage.py` - Comprehensive examples
-   `src/create_dataset_json.py` - Creating JSON datasets
-   `src/test_from_json.py` - Testing from JSON

## Troubleshooting

### Common Issues

1. **JSON files not found**

    - Run `create_dataset_json.py` first
    - Check output directory path

2. **Image loading errors**

    - Check image file integrity
    - Verify relative paths in JSON

3. **Memory issues**

    - Reduce batch size
    - Reduce num_workers

4. **Inconsistent splits**
    - Use same random_seed
    - Don't modify JSON files manually

### Debugging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Best Practices

1. **Always use JSON datasets** for production
2. **Set fixed random_seed** for reproducibility
3. **Version your JSON files** with your code
4. **Backup JSON files** - they're your dataset definition
5. **Validate splits** after creation
6. **Use relative paths** for portability
7. **Document your splits** and rationale

## Performance Notes

-   JSON loading is ~3-5x faster than directory scanning
-   Memory usage is similar between methods
-   First-time JSON creation takes extra time
-   Subsequent loads are much faster
