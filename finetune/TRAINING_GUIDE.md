# Walrus Navier-Stokes Finetuning Guide

This guide explains how to finetune the Walrus foundation model on the Navier-Stokes Spectral dataset.

## Overview

The `train_navier_stokes_finetune.py` script provides a complete training pipeline for finetuning Walrus on novel physics data. It includes:

- Custom PyTorch Dataset for HDF5 Navier-Stokes data
- DataModule wrapper compatible with Walrus trainer
- Pretrained weight loading with field index alignment
- Distributed training support (DDP)
- Automatic mixed precision training
- Checkpointing and visualization
- WandB logging integration

## Prerequisites

1. **Data**: HDF5 file with Navier-Stokes data (as created in `NS_Walrus_ZSL.py`)
2. **Pretrained Model**: Downloaded Walrus checkpoint and config
3. **Environment**: Walrus environment with all dependencies installed

## Quick Start

### Single GPU Training

```bash
python train_navier_stokes_finetune.py
```

### Multi-GPU Training (4 GPUs)

```bash
torchrun --nproc_per_node=4 train_navier_stokes_finetune.py
```

### Multi-Node Training (2 nodes, 4 GPUs each)

```bash
# On node 0:
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
    --master_addr=<node0_ip> --master_port=29500 \
    train_navier_stokes_finetune.py

# On node 1:
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
    --master_addr=<node0_ip> --master_port=29500 \
    train_navier_stokes_finetune.py
```

## Configuration

The main configuration variables are at the top of `train_navier_stokes_finetune.py`:

```python
# Paths
HDF5_DATA_PATH = "path/to/navier_stokes_spectral.hdf5"
CHECKPOINT_PATH = "path/to/walrus.pt"
CONFIG_PATH = "path/to/extended_config.yaml"
EXPERIMENT_ROOT = "./experiments/navier_stokes_finetune"

# Training hyperparameters
BATCH_SIZE = 4
N_STEPS_INPUT = 4          # Number of input timesteps
N_STEPS_OUTPUT = 1         # Number of output timesteps (1 for single-step)
TARGET_SIZE = 128          # Spatial resolution
MAX_EPOCHS = 50
LEARNING_RATE = 1e-5
NUM_WORKERS = 4

# Validation settings
VAL_FREQUENCY = 5          # Validate every N epochs
ROLLOUT_VAL_FREQUENCY = 10 # Rollout validation every N epochs
MAX_ROLLOUT_STEPS = 20
SHORT_VALIDATION_LENGTH = 10

# Optimization settings
ENABLE_AMP = True          # Automatic mixed precision
AMP_TYPE = "float16"
GRAD_ACC_STEPS = 1         # Gradient accumulation steps
CLIP_GRADIENT = 1.0        # Gradient clipping norm
```

## Key Features

### 1. Custom Dataset (`NavierStokesSpectralDataset`)

Loads Navier-Stokes data from HDF5 and formats it for Walrus:

- Handles train/val/test splits
- Resizes spatial dimensions to model-compatible sizes
- Creates proper field mappings to pretrained embeddings
- Generates sliding window samples for efficient training

### 2. Field Mapping

Maps Navier-Stokes fields to pretrained Walrus embeddings:

```python
field_to_index_map = {
    "velocity_x": 4,   # Uses pretrained embedding 4
    "velocity_y": 5,   # Uses pretrained embedding 5
    "velocity_z": 6,   # Padding, uses pretrained embedding 6
    "pressure": 3,     # Uses pretrained embedding 3
}
```

This leverages the foundation model's knowledge of similar physical fields.

### 3. Training Strategy

**Single-step training**: During training, the model predicts one timestep ahead (N_STEPS_OUTPUT=1). This is more stable and faster.

**Autoregressive rollout during validation**: During validation, the model performs multi-step rollouts (up to MAX_ROLLOUT_STEPS) to evaluate long-term prediction accuracy.

**Loss function**: Uses VRMSE (Variance-normalized RMSE) as the primary loss, which normalizes by the variance of each field.

### 4. Learning Rate Schedule

Uses cosine annealing with a minimum learning rate:

```python
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=MAX_EPOCHS,
    eta_min=LEARNING_RATE / 10,
)
```

### 5. Checkpointing

Automatically saves:
- Best model (based on validation loss)
- Last model (end of training)
- Checkpoint every N epochs (configurable)

Checkpoints include:
- Model state dict
- Optimizer state dict
- Epoch number
- Validation loss

## Memory Optimization Tips

If you encounter out-of-memory errors:

1. **Reduce batch size**: Decrease `BATCH_SIZE`
2. **Reduce spatial resolution**: Decrease `TARGET_SIZE` (must be 32, 128, 256, 384, 512, 768, or 1024)
3. **Use gradient accumulation**: Increase `GRAD_ACC_STEPS` (effective batch size = BATCH_SIZE × GRAD_ACC_STEPS)
4. **Reduce rollout steps**: Decrease `MAX_ROLLOUT_STEPS` during validation
5. **Use mixed precision**: Ensure `ENABLE_AMP = True`
6. **Reduce input timesteps**: Decrease `N_STEPS_INPUT` if model allows

## Training Workflow

1. **Initialization**:
   - Load pretrained Walrus checkpoint
   - Align field embeddings to Navier-Stokes fields
   - Setup optimizer, scheduler, and trainer

2. **Training Loop** (for each epoch):
   - Train on full training set with single-step prediction
   - Optionally validate with one-step prediction (every `VAL_FREQUENCY` epochs)
   - Optionally validate with rollout (every `ROLLOUT_VAL_FREQUENCY` epochs)
   - Save checkpoint if validation improves

3. **Validation Types**:
   - **One-step validation**: Fast, evaluates single-step prediction accuracy
   - **Rollout validation**: Slower, evaluates long-term autoregressive rollout

4. **Testing**:
   - After training, evaluates on held-out test set
   - Performs full rollout validation on test data

## Monitoring Training

### WandB Dashboard

If WandB logging is enabled, you can monitor:

- Training loss
- Validation loss (one-step and rollout)
- Per-field metrics (VRMSE for velocity_x, velocity_y, pressure)
- Learning rate
- Gradient norms
- Memory usage
- Training time per batch

### Local Logs

Check the console output for:
- Batch-level training progress
- Validation metrics per dataset
- Checkpoint saving notifications

### Visualizations

The script saves visualizations to `EXPERIMENT_ROOT/visualizations/`:
- Rollout videos (if enabled)
- Error plots over time
- Field-by-field comparisons

## Output Structure

```
experiments/navier_stokes_finetune/
├── checkpoints/
│   ├── best_model.pt
│   ├── last_model.pt
│   └── checkpoint_epoch_*.pt
├── visualizations/
│   ├── navier_stokes_spectral/
│   │   ├── rollout_losses/
│   │   └── videos/
│   └── loss_dicts/
└── extended_config.yaml
```

## Customization

### Using a Different Dataset

To adapt this script to your own PDE data:

1. **Create a custom Dataset class** that loads your data and formats it as:
   ```python
   {
       "input_fields": [B, T_in, H, W, D, C],
       "output_fields": [B, T_out, H, W, D, C],
       "constant_fields": [B, H, W, D, C_const],
       "boundary_conditions": [B, 3, 2],
       "padded_field_mask": [C],
       "field_indices": [C],
       "metadata": WellMetadata,
   }
   ```

2. **Map your fields to pretrained embeddings** in the field_to_index_map. Check the pretrained model's field mapping to find similar fields.

3. **Create appropriate metadata** with correct spatial dimensions, field names, and boundary condition types.

### Advanced Finetuning Strategies

**Freezing layers**: Freeze early layers and only train later layers:

```python
# Freeze embedding layers
for param in model.field_embedding.parameters():
    param.requires_grad = False

# Only train decoder
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LEARNING_RATE,
)
```

**Layer-wise learning rates**: Use different learning rates for different parts:

```python
param_groups = [
    {"params": model.field_embedding.parameters(), "lr": LEARNING_RATE / 10},
    {"params": model.encoder.parameters(), "lr": LEARNING_RATE / 2},
    {"params": model.decoder.parameters(), "lr": LEARNING_RATE},
]
optimizer = torch.optim.AdamW(param_groups)
```

**Curriculum learning**: Start with shorter rollouts and gradually increase:

```python
# In training loop, adjust MAX_ROLLOUT_STEPS based on epoch
MAX_ROLLOUT_STEPS = min(5 + epoch // 5, 20)
```

## Troubleshooting

**Issue**: Model diverges (loss becomes NaN)
- **Solution**: Reduce learning rate, enable gradient clipping, check data for NaN/Inf values

**Issue**: Validation loss doesn't improve
- **Solution**: Increase learning rate, train longer, reduce regularization, check data quality

**Issue**: Out of memory during rollout validation
- **Solution**: Reduce `MAX_ROLLOUT_STEPS`, use batch_size=1 for rollout (already default)

**Issue**: Training is too slow
- **Solution**: Increase batch size, reduce validation frequency, use multiple GPUs, enable AMP

**Issue**: Field embeddings don't align
- **Solution**: Check field_to_index_map, ensure fields exist in pretrained checkpoint, verify field dimensions

## Performance Tips

1. **Use appropriate batch size**: Too small = slow training, too large = poor generalization. Start with batch_size=4-8.

2. **Tune learning rate**: Use learning rate finder or start with 1e-5 for finetuning.

3. **Enable mixed precision**: 2x speedup with minimal accuracy loss.

4. **Profile your code**: Use PyTorch profiler to identify bottlenecks:
   ```python
   with torch.profiler.profile(...) as prof:
       trainer.train_one_epoch(...)
   print(prof.key_averages().table())
   ```

5. **Use multiple workers**: Set NUM_WORKERS to match CPU cores (but not too high, 4-8 is usually good).

## Citation

If you use this training script or Walrus in your research, please cite:

```bibtex
@article{mcccabe2024walrus,
  title={Walrus: A foundation model for physical simulations},
  author={McCabe, Michael and others},
  journal={arXiv preprint arXiv:2410.xxxxx},
  year={2024}
}
```

## Support

For issues or questions:
- Check the [Walrus repository](https://github.com/polymathic-ai/the_walrus)
- Open an issue on GitHub
- Contact the Polymathic AI team
