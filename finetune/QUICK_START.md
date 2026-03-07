# Quick Start: Finetuning Walrus on Navier-Stokes

This is a quick reference guide to get you started with finetuning Walrus on your Navier-Stokes dataset.

## Prerequisites

✅ Navier-Stokes data in HDF5 format (see `NS_Walrus_ZSL.py` for data preparation)
✅ Downloaded Walrus pretrained checkpoint and config
✅ Walrus environment activated
✅ GPU with CUDA available (recommended)

## 3-Step Quickstart

### 1. Edit Paths in Training Script

Open `train_navier_stokes_finetune.py` and update these paths:

```python
HDF5_DATA_PATH = "path/to/your/navier_stokes_spectral.hdf5"
CHECKPOINT_PATH = "path/to/walrus.pt"
CONFIG_PATH = "path/to/extended_config.yaml"
```

### 2. Run Training

**Option A: Single GPU**
```bash
python train_navier_stokes_finetune.py
```

**Option B: Multiple GPUs (e.g., 4 GPUs)**
```bash
torchrun --nproc_per_node=4 train_navier_stokes_finetune.py
```

**Option C: Use the launcher script**
```bash
./launch_finetune.sh --mode single        # Single GPU
./launch_finetune.sh --mode multi-gpu -n 4  # 4 GPUs
```

### 3. Monitor Training

- **Console**: Watch training progress in terminal
- **WandB**: Check your WandB dashboard for metrics
- **Checkpoints**: Find saved models in `experiments/navier_stokes_finetune/checkpoints/`

## Key Hyperparameters

Edit these in the script to tune training:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BATCH_SIZE` | 4 | Number of samples per batch |
| `LEARNING_RATE` | 1e-5 | Initial learning rate |
| `MAX_EPOCHS` | 50 | Total training epochs |
| `N_STEPS_INPUT` | 4 | Input timesteps |
| `N_STEPS_OUTPUT` | 1 | Output timesteps (keep at 1 for training) |
| `TARGET_SIZE` | 128 | Spatial resolution (32, 128, 256, 384, 512, 768, 1024) |
| `VAL_FREQUENCY` | 5 | Validate every N epochs |
| `CLIP_GRADIENT` | 1.0 | Gradient clipping threshold |

## Common Adjustments

### If out of memory:
```python
BATCH_SIZE = 2              # Reduce batch size
TARGET_SIZE = 128           # Use smaller spatial resolution
GRAD_ACC_STEPS = 2          # Gradient accumulation
```

### For faster training:
```python
BATCH_SIZE = 8              # Increase batch size
NUM_WORKERS = 8             # More data loading workers
VAL_FREQUENCY = 10          # Less frequent validation
ENABLE_AMP = True           # Use mixed precision
```

### For better accuracy:
```python
LEARNING_RATE = 5e-6        # Lower learning rate
MAX_EPOCHS = 100            # Train longer
N_STEPS_INPUT = 6           # More context
MAX_ROLLOUT_STEPS = 30      # Longer rollouts in validation
```

## Outputs

After training, you'll find:

```
experiments/navier_stokes_finetune/
├── checkpoints/
│   ├── best_model.pt           # Best model by validation loss
│   └── last_model.pt            # Final model
├── visualizations/
│   └── navier_stokes_spectral/
│       ├── rollout_losses/      # Loss evolution plots
│       └── videos/              # Rollout visualizations
└── extended_config.yaml         # Saved configuration
```

## Loading Trained Model

To use your finetuned model for inference:

```python
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate

# Load config
config = OmegaConf.load("path/to/extended_config.yaml")

# Create model
model = instantiate(config.model, n_states=7)  # Adjust n_states

# Load checkpoint
checkpoint = torch.load("experiments/navier_stokes_finetune/checkpoints/best_model.pt")
model.load_state_dict(checkpoint['model'])
model.eval()

# Use for inference (see NS_Walrus_ZSL.py for full example)
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **NaN loss** | Reduce learning rate, enable gradient clipping |
| **Slow training** | Increase batch size, use multiple GPUs, enable AMP |
| **Poor validation** | Train longer, reduce learning rate, check data quality |
| **Out of memory** | Reduce batch size, reduce spatial resolution, use gradient accumulation |
| **No improvement** | Check learning rate, verify field mappings, inspect data |

## Next Steps

- ✨ **Tune hyperparameters** based on validation metrics
- 📊 **Analyze results** in WandB dashboard
- 🔬 **Test on held-out data** using the test dataloader
- 🚀 **Deploy model** for downstream applications

## Need Help?

- 📖 Read the full [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
- 💬 Check [Walrus GitHub issues](https://github.com/polymathic-ai/the_walrus/issues)
- 📧 Contact the Polymathic AI team

---

**Happy Finetuning! 🎉**
