# Cat Breed and Emotion Classification

A modular, concise PyTorch pipeline for cat classification supporting both single-task (breed or emotion) and multitask learning with configurable backbones and JSON dataset formats.

## Features

-   **Multiple task modes**: Breed classification, emotion classification, or joint multitask learning
-   **JSON dataset support**: Flexible dataset format with metadata and labels
-   **Configurable architectures**: MobileNetV2, MobileNetV3, and other torchvision backbones
-   **Pre-trained ImageNet weights** for fast convergence and transfer learning
-   **Advanced training features**:
    -   Data augmentation with albumentations
    -   Early stopping and learning rate scheduling
    -   Progress tracking with tqdm
    -   TensorBoard integration for experiment tracking
-   **Comprehensive evaluation**: Metrics, confusion matrices, and visualizations
-   **Real-time inference**: Support for images, videos, and webcam feed
-   **YAML configuration**: Full configuration via files with CLI override support
-   **Robust checkpointing**: Automatic model saving and experiment reproducibility

## Project Structure

```
├── config/                     # Legacy configuration files
├── data/                       # Dataset directory
│   ├── data-breed/             # Cat breed dataset
│   │   └── processed/          # Organized breed folders
│   ├── data-emo/               # Cat emotion dataset
│   │   └── processed/          # Organized emotion folders
│   └── json_datasets_*/        # JSON dataset files
├── docs/                       # Documentation and guides
├── notebooks/                  # Jupyter notebooks for analysis
├── src/                        # Main source code
│   ├── configs/                # YAML configuration files
│   │   ├── breed_json_config.yaml
│   │   ├── emotion_json_config.yaml
│   │   └── multitask_json_config.yaml
│   ├── checkpoints/            # Model checkpoints and logs
│   ├── utils/                  # Utility modules
│   │   ├── data_utils.py       # Data loading and JSON dataset handling
│   │   └── visualization.py    # Plotting and visualization utilities
│   ├── model.py                # Model architectures and factory functions
│   ├── trainer.py              # Single-task training loop
│   ├── multitask_trainer.py    # Multitask training with joint losses
│   ├── evaluate.py             # Model evaluation and metrics
│   ├── main.py                 # CLI entry point and argument parsing
│   └── test.py                 # Inference and prediction utilities
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/username/cat-breed-classification.git
    cd cat-breed-classification
    ```

2. Create and activate a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset Preparation

The system supports two dataset formats:

### 1. Folder-based Structure (Traditional)

```
data/data-breed/processed/
├── Abyssinian/
│   ├── Abyssinian_1.jpg
│   ├── Abyssinian_2.jpg
│   └── ...
├── Bengal/
│   ├── Bengal_1.jpg
│   └── ...
└── ...
```

### 2. JSON Dataset Format (Recommended)

Create JSON files with image paths and labels:

```json
{
  "images": [
    {
      "file_path": "Abyssinian/img_001.jpg",
      "breed_label": "Abyssinian",
      "emotion_label": "happy"
    },
    {
      "file_path": "Bengal/img_002.jpg",
      "breed_label": "Bengal",
      "emotion_label": "calm"
    }
  ],
  "breed_classes": ["Abyssinian", "Bengal", ...],
  "emotion_classes": ["angry", "happy", "sad", "other"]
}
```

JSON datasets enable multitask learning and flexible data organization.

## Usage

### Training

**Single-task training** (breed classification only):

```bash
python src/main.py train --mode breed --config-path src/configs/breed_json_config.yaml
```

**Emotion classification**:

```bash
python src/main.py train --mode emotion --config-path src/configs/emotion_json_config.yaml
```

**Multitask training** (joint breed and emotion classification):

```bash
python src/main.py train --mode multitask --config-path src/configs/multitask_json_config.yaml
```

**Custom training parameters**:

```bash
python src/main.py train --mode multitask \
    --backbone mobilenet_v3_large \
    --batch-size 64 \
    --epochs 50 \
    --lr 0.0005 \
    --train-json data/json_datasets/train.json \
    --val-json data/json_datasets/val.json \
    --test-json data/json_datasets/test.json
```

**Configuration file only** (no CLI arguments needed):

```bash
python src/main.py train --config-path src/configs/multitask_json_config.yaml
```

### Evaluation

**Evaluate a specific model**:

```bash
python src/main.py evaluate --model-path src/checkpoints/multitask_mobilenet_v2_20250627_143021/best_state.pth
```

**Evaluate latest checkpoint automatically**:

```bash
python src/main.py evaluate --mode multitask
```

**Evaluate with custom test dataset**:

```bash
python src/main.py evaluate --test-json data/json_datasets/test.json --data-root data/data-breed/processed
```

### Prediction

**Single image inference**:

```bash
python src/main.py predict --input path/to/cat_image.jpg --model-path src/checkpoints/best_model/best_state.pth
```

**Multitask prediction** (predicts both breed and emotion):

```bash
python src/main.py predict --input path/to/cat_image.jpg --mode multitask
```

### Video Processing

**Process video with multitask model**:

```bash
python src/main.py video --input path/to/video.mp4 --output path/to/output.mp4 --mode multitask
```

### Webcam

**Real-time inference** (shows both breed and emotion):

```bash
python src/main.py webcam --mode multitask
```

## Configuration Files

The system supports YAML configuration files for all parameters. CLI arguments override config values when provided.

### Example Multitask Configuration

```yaml
# Model Configuration
backbone: "mobilenet_v2"
mode: "multitask"

# Training Parameters
batch_size: 32
learning_rate: 3e-4
weight_decay: 1e-4
epochs: 50
early_stopping: 10

# Model Parameters
img_size: 224
shared_features: true

# Multitask Loss Weights
breed_weight: 1.0
emotion_weight: 1.0

# Data Configuration - JSON Datasets
train_json: "data/json_datasets/train_dataset.json"
val_json: "data/json_datasets/val_dataset.json"
test_json: "data/json_datasets/test_dataset.json"
data_root: "data/data-breed/processed"

# Training Configuration
num_workers: 4
device: "cuda"
checkpoint_dir: "src/checkpoints"
use_tensorboard: true
```

### Available Backbones

-   `mobilenet_v2` (default)
-   `mobilenet_v3_small`
-   `mobilenet_v3_large`
-   `efficientnet_b0` through `efficientnet_b7`
-   `resnet18`, `resnet34`, `resnet50`, `resnet101`
-   Any backbone from `torchvision.models`

### Task Modes

-   **`breed`**: Cat breed classification only
-   **`emotion`**: Cat emotion classification only
-   **`multitask`**: Joint breed and emotion classification

For detailed configuration options, see [Configuration Guide](docs/configuration.md).

## Advanced Features

### Multitask Learning

The pipeline supports joint training on multiple tasks (breed + emotion) with:

-   Shared backbone features or task-specific feature extractors
-   Configurable loss weights for balancing tasks
-   Independent evaluation metrics for each task
-   Progress tracking with tqdm for both tasks

### JSON Dataset Format

Flexible dataset organization with:

-   Metadata support for images
-   Multiple label types per image
-   Easy train/validation/test splits
-   Support for missing labels (single-task subsets)

### Experiment Tracking

-   **TensorBoard integration**: Real-time loss and accuracy plots
-   **Automatic checkpointing**: Best and latest model states
-   **Training curves**: Saved plots for each experiment
-   **Reproducible configs**: YAML files saved with each checkpoint

## License

This project is licensed under the MIT License - see the LICENSE file for details.
