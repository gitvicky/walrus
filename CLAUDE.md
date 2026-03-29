# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Walrus is a cross-domain foundation model for continuum dynamics (fluid simulations, PDEs) built by Polymathic AI. It uses an encoder-processor-decoder architecture trained on 19 physical scenarios from The Well dataset.

## Common Commands

### Installation
```bash
pip install .                    # Basic install
pip install .[test]              # With test dependencies (pytest, ruff, mypy)
pip install .[external_models]   # With external model dependencies
```

### Testing
```bash
pytest tests                     # Run all tests
pytest tests/test_trainer.py     # Run single test file
pytest tests -k "test_name"      # Run tests matching pattern
```

### Linting
```bash
ruff check walrus tests          # Lint code
ruff check --select I walrus tests  # Check import sorting
ruff check --fix walrus tests    # Auto-fix issues
```

### Type Checking
```bash
mypy walrus tests --install-types --non-interactive
```

### Training (Local)
```bash
cd walrus
python train.py server=local distribution=local data=active_matter name=MyExperiment
```

### Training (Distributed via torchrun)
```bash
torchrun --nproc_per_node=4 train.py server=local distribution=local data=active_matter
```

## Architecture

### Directory Structure
- `walrus/` - Main package
  - `train.py` - Main training entry point (uses Hydra)
  - `configs/` - Hydra YAML configuration hierarchy
  - `models/` - Model implementations
  - `data/` - Dataset and dataloader implementations
  - `trainer/` - Training loop, checkpointing, normalization
  - `optim/` - Optimizers (including distributed shampoo)
  - `utils/` - Distribution and experiment utilities
- `tests/` - Pytest test suite
- `finetune/` - Finetuning scripts and guides
- `zero_shot/` - Zero-shot inference examples
- `demo_notebooks/` - Jupyter tutorials

### Model Architecture (IsotropicModel)
The main model in `walrus/models/isotropic_model.py` uses encoder-processor-decoder:

1. **Encoder** (`models/encoders/`): vstride_encoder with hMLP using stride modulation for adaptive downsampling
2. **Processor** (`models/spatiotemporal_blocks/`): Stack of SpaceTimeSplitBlocks with factorized spatial and temporal attention
3. **Decoder** (`models/decoders/`): vstride_decoder (transposed hMLP)

Key features:
- Patch jittering for stability (`models/shared_utils/patch_jitterers.py`)
- Handles 2D and 3D data (pads 2D to 3D internally)
- RMSGroupNorm normalization
- Gradient checkpointing support

### Configuration (Hydra)
Configs are composed from `walrus/configs/`:
- `config.yaml` - Base config with defaults
- `model/` - Model configs (isotropic_model.yaml is primary)
- `data/` - Dataset configs (one per physics scenario)
- `trainer/` - Training settings
- `optimizer/`, `lr_scheduler/` - Optimization
- `distribution/` - DDP/FSDP/HSDP settings
- `server/` - Environment-specific paths

Override via command line: `python train.py model.hidden_dim=1024 trainer.max_epoch=100`

### Data Pipeline
- Uses The Well dataset format (HDF5)
- `data/multidataset.py` - Multi-source dataset handling
- `data/multidatamodule.py` - DataModule for mixed datasets
- `data/inflated_dataset.py` - Padding/inflation for uniform shapes
- `data/mixed_dset_sampler.py` - Efficient heterogeneous sampling

### Trainer
- `trainer/training.py` - Main Trainer class with train/validation loops
- `trainer/checkpoints.py` - Checkpoint loading/saving with field alignment
- `trainer/normalization_strat.py` - Normalization strategies

## Key Concepts

**Field-to-Index Mapping**: Physical fields (velocity_x, pressure, etc.) map to embedding indices. When finetuning on new data, align your field names to pretrained embeddings via `field_to_index_map`.

**Boundary Conditions**: Handled via `the_well.data.datasets.BoundaryCondition` enum. Model adjusts behavior (periodic rolling vs padding) based on BC type.

**Stride Modulation**: Encoder/decoder dynamically adjust downsampling stride to maintain consistent internal token counts across different input resolutions.

## Pre-commit Hooks
The repo uses pre-commit with ruff (linting + import sorting) and mypy. Run `pre-commit install` after cloning.
