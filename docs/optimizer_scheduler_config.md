# Optimizer and Scheduler Configuration

This document describes how to configure optimizers and schedulers in the cat classification system through YAML configuration files.

## Optimizer Options (Config-file only)

The following optimizers are supported:

-   `adamw` (default): AdamW optimizer with weight decay separated from the learning rate
-   `adam`: Adam optimizer
-   `sgd`: Stochastic Gradient Descent with momentum

Example configuration:

```yaml
optimizer: "adam" # Choose from "adamw", "adam", or "sgd"
learning_rate: 0.001
weight_decay: 0.0001
```

## Scheduler Options (Config-file only)

The following learning rate schedulers are supported:

-   `reduce_lr_on_plateau` (default): Reduces learning rate when a metric plateaus
-   `cosine`: Cosine annealing scheduler
-   `step`: Step scheduler that decreases the learning rate at fixed intervals
-   `none`: No learning rate scheduling

### ReduceLROnPlateau Configuration

```yaml
scheduler: "reduce_lr_on_plateau"
scheduler_patience: 5 # Number of epochs with no improvement before reducing LR
scheduler_factor: 0.2 # Factor by which to reduce learning rate (e.g., 0.2 = reduce by 80%)
scheduler_min_lr: 1e-6 # Minimum learning rate
```

### Cosine Annealing Configuration

```yaml
scheduler: "cosine"
scheduler_min_lr: 1e-6 # Minimum learning rate
scheduler_t_max: 20 # Number of epochs for a complete cosine cycle
```

### Step Scheduler Configuration

```yaml
scheduler: "step"
scheduler_step_size: 15 # Epochs between LR reductions
scheduler_gamma: 0.1 # Multiplicative factor for LR reduction
```

## Example Full Configuration

```yaml
# Model configuration
backbone: "mobilenet_v2"
img_size: 224
mode: "breed"

# Dataset configuration
data_root: "../../data"
train_json: "../../data/json_breed_datasets/train_dataset.json"
val_json: "../../data/json_breed_datasets/val_dataset.json"
test_json: "../../data/json_breed_datasets/test_dataset.json"

# Training parameters
batch_size: 32
epochs: 50
early_stopping: 10

# Optimizer configuration (config-file only)
optimizer: "adam"
learning_rate: 0.001
weight_decay: 0.0001

# Scheduler configuration (config-file only)
scheduler: "cosine"
scheduler_min_lr: 0.000001
scheduler_t_max: 20
```

## Usage

To use these optimizer and scheduler configurations, simply provide a YAML config file:

```bash
python src/main.py train --config-path configs/optimizer_config_example.yaml
```

Unlike other parameters, optimizer and scheduler settings are only configurable through YAML config files and don't have corresponding command-line arguments.
