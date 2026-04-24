# Copilot Instructions for Walrus

## Build, test, lint
- Install package: `pip install .`
- Install with test deps: `pip install .[test]`
- Run full test suite: `pytest tests`
- Run a single test file: `pytest tests/test_trainer.py`
- Run tests by name: `pytest tests -k "test_name"`
- Lint: `ruff check walrus tests`
- Import sorting check: `ruff check --select I walrus tests`
- Type check: `mypy walrus tests --install-types --non-interactive`

## High-level architecture
- Entry point is `walrus/train.py`, which uses Hydra configs from `walrus/configs/` to compose model, data, trainer, optimizer, distribution, and server settings.
- The main model (`walrus/models/isotropic_model.py`) is an encoder-processor-decoder: vstride encoder/decoder with stride modulation and a processor stack of SpaceTimeSplitBlocks for factorized spatial/temporal attention.
- Data pipeline targets The Well HDF5 format with multi-source sampling and shape normalization handled by `multidataset.py`, `multidatamodule.py`, `inflated_dataset.py`, and `mixed_dset_sampler.py`.
- Training loops and checkpointing live in `walrus/trainer/`, with normalization strategies in `trainer/normalization_strat.py`.

## Key conventions
- Physical field names map to embedding indices; when finetuning, align fields via `field_to_index_map` to reuse pretrained embeddings.
- Boundary conditions follow `the_well.data.datasets.BoundaryCondition`; periodic cases use rolling while others use padding behavior.
- 2D inputs are padded to 3D internally so most model code assumes 3D tensors.
- Stride modulation keeps internal token counts consistent across resolutions; patch jittering is used to stabilize long rollouts.
