# Walrus Development Status

## Overview

Walrus is a cross-domain foundation model for continuum dynamics, developed by Polymathic AI and trained on 19 physical scenarios spanning 63 variables in 2D and 3D. This document summarises the current state of our development work building on top of the pretrained model.

## Current Focus: Zero-Shot Learning

The primary active area of development is **zero-shot inference** — using the pretrained Walrus model to predict unseen physics domains without any finetuning.

### Completed Work

Two zero-shot learning pipelines have been tested and validated:

- **Navier-Stokes** (`zero_shot/NS_Walrus_ZSL.py`) — Zero-shot rollout predictions on spectral Navier-Stokes data. Loads HDF5 simulation data, maps velocity and pressure fields to pretrained embeddings, and performs autoregressive multi-step predictions.

- **Shear Flow** (`zero_shot/ShearFlow_Walrus_ZSL.py`) — Zero-shot predictions on shear flow data following the same pipeline.

Both scripts produce spatial field comparisons, temporal error evolution, scatter plots, and per-field metrics (MSE, RMSE, MAE, Relative L2, R²).

### Zero-Shot Pipeline Summary

1. Load simulation data from HDF5 files
2. Convert to Walrus batch format with field-to-embedding index mapping (e.g. `velocity_x → 4`, `velocity_y → 5`, `pressure → 3`)
3. Construct boundary conditions and padding masks (velocity_z used as padding for 2D data)
4. Load pretrained checkpoint and align field embeddings
5. Run autoregressive rollout with RevIN normalisation
6. Evaluate and visualise results

### Key Observations

- Field mapping is critical — correctly aligning new domain fields to the pretrained embedding indices determines prediction quality.
- 2D simulations require padding to accommodate the 3D-capable architecture (velocity_z serves as a padded field).
- RevIN (reversible instance normalisation) handles per-sample statistics for transfer to unseen distributions.

## Finetune Pipeline (Untested)

A complete finetuning pipeline has been set up but **has not yet been run or validated**. All infrastructure is in place under `finetune/`.

### Components

| File | Purpose |
|------|---------|
| `train_navier_stokes_finetune.py` | Full training script with dataset, datamodule, and training loop |
| `launch_finetune.sh` | Launcher for single and multi-GPU training via torchrun |
| `QUICK_START.md` | 3-step quickstart reference |
| `TRAINING_GUIDE.md` | Comprehensive guide with hyperparameter recommendations |

### Pipeline Design

- **Dataset**: `NavierStokesSpectralDataset` — loads HDF5 data with configurable train/val/test splits (70/15/15 default), sliding window sampling, and automatic spatial resizing.
- **Training**: Single-step prediction training with multi-step rollout validation. AdamW optimiser with cosine annealing, learning rate 1e-5, VRMSE loss.
- **Features**: AMP mixed precision, gradient accumulation, WandB logging, best-model checkpointing, rollout visualisation.
- **Multi-GPU**: Supported via torchrun and DistributedDataParallel.

### What Remains

- End-to-end training run to validate the pipeline works
- Hyperparameter tuning (batch size, learning rate, number of epochs)
- Comparison of finetuned vs zero-shot performance
- Testing on additional physical domains beyond Navier-Stokes

## Project Structure

```
walrus/
├── walrus/              # Core library (model, data, trainer, configs)
├── zero_shot/           # Zero-shot learning scripts (active development)
├── finetune/            # Finetuning pipeline (implemented, untested)
├── demo_notebooks/      # Tutorial notebooks and example data
└── tests/               # Unit and integration tests
```

## Model Architecture

The pretrained model uses an encoder-processor-decoder structure:

- **Encoder**: Adaptive stride encoding (vstride) with variable downsampling
- **Processor**: 8 repeating blocks with factorised space-time attention
- **Decoder**: Adaptive stride decoding mirroring the encoder
- **Hidden dim**: 768, with RMSGroupNorm and patch jittering for stability

## Dependencies

Core: PyTorch (>=2.5.1), Hydra, einops, the_well (Polymathic AI data framework), h5py, wandb.

## Next Steps

1. Run and validate the finetuning pipeline end-to-end
2. Benchmark finetuned model against zero-shot baselines
3. Extend zero-shot evaluation to additional physics domains
4. Investigate sensitivity to field mapping choices and boundary condition specification
