"""
Walrus Zero-Shot Learning (ZSL) Framework
==========================================

Generic framework for performing zero-shot inference with the pretrained
Walrus foundation model on arbitrary PDE datasets.

Usage
-----
1. Write a data extractor function for your dataset (see examples/ directory).
2. Create a ZSLConfig pointing to your checkpoint, config, and data.
3. Call ``run_zsl`` with your extractor and config.

Example
-------
    from walrus_zsl import run_zsl, ZSLConfig
    from extractors.my_extractor import extract

    config = ZSLConfig(
        checkpoint_path="checkpoints/walrus.pt",
        checkpoint_config_path="configs/extended_config.yaml",
        dataset_name="my_simulation",
    )
    results = run_zsl(extract, config)

See ``FIELD_INDEX_REFERENCE`` below for the complete list of pretrained
field embeddings, or inspect the YAML source at:
    walrus/configs/data/field_index_map_override/full_well_field_index.yaml
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from matplotlib.gridspec import GridSpec
from omegaconf import OmegaConf
from the_well.data.datasets import WellMetadata

from walrus.data.well_to_multi_transformer import ChannelsFirstWithTimeFormatter
from walrus.trainer.normalization_strat import SamplewiseRevNormalization
from walrus.trainer.training import expand_mask_to_match
from walrus.utils.experiment_utils import align_checkpoint_with_field_to_index_map

logger = logging.getLogger(__name__)


# =============================================================================
# PRETRAINED FIELD INDEX REFERENCE
# =============================================================================
# This is the complete map of field names to embedding indices in the
# pretrained Walrus checkpoint. When writing a data extractor, map each of
# your physical variables to the closest match below.
#
# Source: walrus/configs/data/field_index_map_override/full_well_field_index.yaml
# =============================================================================

FIELD_INDEX_REFERENCE: Dict[str, int] = {
    # --- Utility / boundary fields ---
    "closed_boundary":      0,
    "open_boundary":        1,
    "bias_correction":      2,
    # --- Core fluid fields ---
    "pressure":             3,
    "velocity_x":           4,
    "velocity_y":           5,
    "velocity_z":           6,
    "zeros_like_density":   7,
    "speed_of_sound":       8,
    "concentration":        9,
    # --- Rank-2 tensor D (9 components) ---
    "D_xx": 10, "D_xy": 11, "D_xz": 12,
    "D_yx": 13, "D_yy": 14, "D_yz": 15,
    "D_zx": 16, "D_zy": 17, "D_zz": 18,
    # --- Rank-2 tensor E (9 components) ---
    "E_xx": 19, "E_xy": 20, "E_xz": 21,
    "E_yx": 22, "E_yy": 23, "E_yz": 24,
    "E_zx": 25, "E_zy": 26, "E_zz": 27,
    # --- Thermodynamic / conservation fields ---
    "density":              28,
    "energy":               29,
    # --- Polar / spherical velocity ---
    "velocity_r":           30,
    "velocity_theta":       31,
    "velocity_phi":         32,
    # --- Momentum ---
    "momentum_x":           33,
    "momentum_y":           34,
    "momentum_z":           35,
    # --- Complex pressure ---
    "pressure_re":          36,
    "pressure_im":          37,
    # --- Mask ---
    "mask":                 38,
    # --- Magnetic field (Cartesian) ---
    "magnetic_field_x":     39,
    "magnetic_field_y":     40,
    "magnetic_field_z":     41,
    # --- Generic fields ---
    "A":                    42,
    "B":                    43,
    # --- Scalar fields ---
    "height":               44,
    "internal_energy":      45,
    "temperature":          46,
    "electron_fraction":    47,
    "entropy":              48,
    # --- Magnetic field (log-polar / spherical) ---
    "magnetic_field_log_r": 49,
    "magnetic_field_theta": 50,
    "magnetic_field_phi":   51,
    "velocity_log_r":       52,
    # --- Passive scalars ---
    "buoyancy":             53,
    "tracer":               54,
    # --- Log-transformed fields ---
    "log10_density":        55,
    "log10_temperature":    56,
    # --- Rank-2 tensor C (lower-case c_zz + uppercase block) ---
    "c_zz":                 57,
    "C_xx": 58, "C_xy": 59, "C_xz": 60,
    "C_yx": 61, "C_yy": 62, "C_yz": 63,
    "C_zx": 64, "C_zy": 65, "C_zz": 66,
}

# Boundary condition codes used in the [B, ndims, 2] tensor.
BC_WALL = 0
BC_OPEN = 1
BC_PERIODIC = 2


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class FieldSpec:
    """Describes one physical field/channel in the dataset.

    Parameters
    ----------
    name : str
        Human-readable name (e.g. ``"velocity_x"``).
    walrus_index : int
        Index into the pretrained embedding table. Choose from
        ``FIELD_INDEX_REFERENCE``.
    is_padding : bool
        Set ``True`` if this channel is a zero-padding placeholder
        (e.g. ``velocity_z`` in a 2D simulation). The channel will be
        zeroed out after every rollout step.
    tensor_rank : int
        0 for scalars, 1 for vector components, 2 for rank-2 tensor
        components. Used to build ``WellMetadata.field_names``.
    """
    name: str
    walrus_index: int
    is_padding: bool = False
    tensor_rank: int = 0


@dataclass
class ExtractedData:
    """Standardised output returned by a user-written data extractor.

    All tensors use ``float32`` and are on CPU. The framework handles
    device transfer.

    Parameters
    ----------
    fields : torch.Tensor
        Full time-series of shape ``[Nt, H, W, C]`` (2D) or
        ``[Nt, H, W, D, C]`` (3D). Channel ordering must match
        ``field_specs``.
    field_specs : list[FieldSpec]
        One entry per channel, in the same order as the C dimension.
    n_spatial_dims : int
        2 or 3 (native dimensionality *before* any Walrus padding).
    boundary_conditions : list[tuple[int, int]]
        One ``(lower_bc, upper_bc)`` pair per spatial dimension, using
        ``BC_WALL``, ``BC_OPEN``, or ``BC_PERIODIC``.
    dataset_name : str
        Descriptive name for logging / metadata.
    target_resolution : int or None
        If set, spatial dimensions are resized to this value. Must be a
        Walrus-compatible size (see ``VALID_SPATIAL_SIZES``).
    constant_fields : torch.Tensor or None
        Optional constant (time-invariant) fields of shape
        ``[H, W, C_const]`` (2D) or ``[H, W, D, C_const]`` (3D).
    constant_field_names : dict
        Constant field names grouped by tensor rank, same format as
        ``WellMetadata.constant_field_names``. Defaults to empty.
    """
    fields: torch.Tensor
    field_specs: List[FieldSpec]
    n_spatial_dims: int
    boundary_conditions: List[Tuple[int, int]]
    dataset_name: str
    target_resolution: Optional[int] = None
    constant_fields: Optional[torch.Tensor] = None
    constant_field_names: Dict[int, list] = field(
        default_factory=lambda: {0: [], 1: [], 2: []}
    )


@dataclass
class ZSLConfig:
    """Configuration for a zero-shot learning run.

    Parameters
    ----------
    checkpoint_path : str
        Path to the pretrained ``walrus.pt`` checkpoint.
    checkpoint_config_path : str
        Path to the corresponding ``extended_config.yaml``.
    T_in : int
        Number of input (conditioning) timesteps.
    T_out : int
        Number of output timesteps to predict.
    max_rollout_steps : int
        Maximum autoregressive rollout steps.
    trajectory_index : int
        Which trajectory to use if the dataset contains multiple.
    output_dir : str
        Directory for saving plots and results.
    model_epsilon : float
        Epsilon for RevIN normalisation stability.
    target_resolution : int or None
        Optional square resize target for 2D inputs.
    """
    checkpoint_path: str = "demo_notebooks/checkpoints/walrus.pt"
    checkpoint_config_path: str = "demo_notebooks/configs/extended_config.yaml"
    T_in: int = 4
    T_out: int = 20
    max_rollout_steps: int = 200
    trajectory_index: int = 0
    output_dir: str = "zero_shot/results"
    model_epsilon: float = 1e-5
    target_resolution: Optional[int] = None


@dataclass
class ZSLResults:
    """Container for zero-shot learning results."""
    predictions: torch.Tensor       # [B, T, H, W, (D), C] real fields only
    references: torch.Tensor        # [B, T, H, W, (D), C] real fields only
    predictions_all: torch.Tensor   # [B, T, H, W, (D), C] all channels
    references_all: torch.Tensor    # [B, T, H, W, (D), C] all channels
    real_field_names: List[str]
    all_field_names: List[str]
    metrics: Dict[str, Dict[str, float]]
    config: ZSLConfig


# Valid spatial sizes (size // 32 must be in {0, 1, 4, 8, 12, 16, 24, 32}).
VALID_SPATIAL_SIZES = [32, 128, 256, 384, 512, 768, 1024]


# Type alias for extractor functions.
Extractor = Callable[[ZSLConfig], ExtractedData]


# =============================================================================
# Internal helpers
# =============================================================================

def _resize_fields(fields: torch.Tensor, target_size: int) -> torch.Tensor:
    """Bilinear resize of ``[Nt, H, W, C]`` → ``[Nt, target, target, C]``."""
    # F.interpolate expects [N, C, H, W]
    Nt, H, W, C = fields.shape
    x = fields.permute(0, 3, 1, 2)  # [Nt, C, H, W]
    x = F.interpolate(x, size=(target_size, target_size), mode="bilinear", align_corners=False)
    return x.permute(0, 2, 3, 1)    # [Nt, H, W, C]


def _build_metadata(data: ExtractedData, spatial_res: tuple) -> WellMetadata:
    """Build a ``WellMetadata`` object from extracted data."""
    # Group field names by tensor rank
    field_names: Dict[int, list] = {0: [], 1: [], 2: []}
    for spec in data.field_specs:
        field_names[spec.tensor_rank].append(spec.name)

    # Walrus always works in 3D internally; n_spatial_dims in metadata
    # controls the normalization reduction dimensions.
    # If native 2D: we pad D=1 later, but can report either 2 or 3 here
    # depending on whether we pad the BC tensor to 3 dims.
    n_dims_for_metadata = data.n_spatial_dims

    return WellMetadata(
        dataset_name=data.dataset_name,
        n_spatial_dims=n_dims_for_metadata,
        field_names=field_names,
        spatial_resolution=spatial_res,
        scalar_names=[],
        constant_field_names=data.constant_field_names,
        constant_scalar_names=[],
        boundary_condition_types=[],
        n_files=[],
        n_trajectories_per_file=[],
        n_steps_per_trajectory=[],
        grid_type="cartesian",
    )


def _build_batch(
    data: ExtractedData,
    config: ZSLConfig,
    device: torch.device,
) -> dict:
    """Convert ``ExtractedData`` into a Walrus batch dictionary."""
    fields = data.fields  # [Nt, H, W, C] or [Nt, H, W, D, C]

    # --- Spatial resize (2D only for now) ---
    needs_resize = data.target_resolution is not None
    if needs_resize and data.n_spatial_dims == 2:
        assert len(fields.shape) == 4, (
            f"Expected [Nt, H, W, C] for 2D data, got {fields.shape}"
        )
        target = data.target_resolution
        if target not in VALID_SPATIAL_SIZES:
            logger.warning(
                f"target_resolution={target} is not in VALID_SPATIAL_SIZES "
                f"{VALID_SPATIAL_SIZES}. Model may fail."
            )
        fields = _resize_fields(fields, target)
        logger.info(f"Resized spatial dims to {target}x{target}")

    # --- Add depth dimension for 2D data ---
    if data.n_spatial_dims == 2 and len(fields.shape) == 4:
        fields = fields.unsqueeze(-2)  # [Nt, H, W, 1, C]

    # Determine spatial resolution from the (possibly resized) data
    if data.n_spatial_dims == 2:
        Nx, Ny = fields.shape[1], fields.shape[2]
        D = 1
        spatial_res = (Nx, Ny)
    else:
        Nx, Ny, D = fields.shape[1], fields.shape[2], fields.shape[3]
        spatial_res = (Nx, Ny, D)

    # --- Split into input / output windows ---
    T_in = config.T_in
    T_out = min(config.T_out, fields.shape[0] - T_in)
    input_fields = fields[:T_in].unsqueeze(0)                   # [1, T_in, H, W, D, C]
    output_fields = fields[T_in : T_in + T_out].unsqueeze(0)    # [1, T_out, H, W, D, C]

    # --- Field indices and padding mask ---
    field_indices = torch.tensor([s.walrus_index for s in data.field_specs])
    padded_field_mask = torch.tensor([not s.is_padding for s in data.field_specs])

    # --- Boundary conditions: [B, ndims, 2] ---
    bc_tensor = torch.tensor([[list(bc) for bc in data.boundary_conditions]])

    # --- Constant fields ---
    if data.constant_fields is not None:
        const = data.constant_fields
        if data.n_spatial_dims == 2 and len(const.shape) == 3:
            const = const.unsqueeze(-2)  # [H, W, 1, C_const]
        constant_fields = const.unsqueeze(0).to(device)  # [1, H, W, D, C_const]
    else:
        constant_fields = torch.empty(1, Nx, Ny, D, 0, device=device)

    # --- Metadata ---
    metadata = _build_metadata(data, spatial_res)

    batch = {
        "input_fields": input_fields.to(device),
        "output_fields": output_fields.to(device),
        "constant_fields": constant_fields,
        "boundary_conditions": bc_tensor.to(device),
        "padded_field_mask": padded_field_mask.to(device),
        "field_indices": field_indices.to(device),
        "metadata": metadata,
    }

    return batch


def _load_model(config: ZSLConfig, device: torch.device):
    """Load the pretrained Walrus model and align field embeddings."""
    checkpoint_raw = torch.load(
        config.checkpoint_path, map_location="cpu", weights_only=True
    )["app"]["model"]
    cfg = OmegaConf.load(config.checkpoint_config_path)

    field_to_index_map = dict(cfg.data.field_index_map_override)
    new_field_to_index_map = dict(field_to_index_map)

    model = instantiate(
        cfg.model,
        n_states=max(new_field_to_index_map.values()) + 1,
    )

    revised_checkpoint = align_checkpoint_with_field_to_index_map(
        checkpoint_state_dict=checkpoint_raw,
        model_state_dict=model.state_dict(),
        checkpoint_field_to_index_map=field_to_index_map,
        model_field_to_index_map=new_field_to_index_map,
    )

    model.load_state_dict(revised_checkpoint)
    model.to(device)
    model.eval()

    # Build RevIN normalizer from checkpoint config
    revin = instantiate(cfg.trainer.revin)()

    return model, revin, cfg


# =============================================================================
# Rollout
# =============================================================================

def rollout(
    model,
    revin,
    batch: dict,
    formatter: ChannelsFirstWithTimeFormatter,
    max_rollout_steps: int = 200,
    model_epsilon: float = 1e-5,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run autoregressive rollout and return (predictions, references).

    Returns tensors in Well format: ``[B, T, H, W, (D), C]``.
    """
    metadata = batch["metadata"]

    batch = {
        k: v.to(device) if k not in {"metadata", "boundary_conditions"} else v
        for k, v in batch.items()
    }

    # Check for mask in constant fields
    if "mask" in metadata.constant_field_names[0]:
        mask_index = metadata.constant_field_names[0].index("mask")
        mask = batch["constant_fields"][..., mask_index : mask_index + 1]
        mask = mask.to(device, dtype=torch.bool)
    else:
        mask = None

    inputs, y_ref = formatter.process_input(
        batch, causal_in_time=model.causal_in_time, predict_delta=True, train=False
    )

    T_in = batch["input_fields"].shape[1]
    if model.causal_in_time:
        effective_max = max_rollout_steps + (T_in - 1)
    else:
        effective_max = max_rollout_steps
    rollout_steps = min(y_ref.shape[1], effective_max)
    train_rollout_limit = 1

    y_ref = y_ref[:, :rollout_steps]
    moving_batch = copy.deepcopy(batch)
    y_preds: list = []

    for i in range(train_rollout_limit - 1, rollout_steps):
        inputs, _ = formatter.process_input(moving_batch)
        inputs = list(inputs)

        with torch.no_grad():
            normalization_stats = revin.compute_stats(
                inputs[0], metadata, epsilon=model_epsilon
            )

        normalized_inputs = inputs[:]
        normalized_inputs[0] = revin.normalize_stdmean(
            normalized_inputs[0], normalization_stats
        )

        y_pred = model(
            normalized_inputs[0],
            normalized_inputs[1],
            normalized_inputs[2].tolist(),
            metadata=metadata,
        )

        if model.causal_in_time:
            y_pred = y_pred[-1:]

        y_pred = inputs[0][-y_pred.shape[0] :].float() + revin.denormalize_delta(
            y_pred, normalization_stats
        )

        y_pred = formatter.process_output(y_pred, metadata)[..., : y_ref.shape[-1]]

        if mask is not None:
            mask_pred = expand_mask_to_match(mask, y_pred)
            y_pred.masked_fill_(mask_pred, 0)

        y_pred = y_pred.masked_fill(~batch["padded_field_mask"], 0.0)

        if i != rollout_steps - 1:
            moving_batch["input_fields"] = torch.cat(
                [moving_batch["input_fields"][:, 1:], y_pred[:, -1:]], dim=1
            )

        if model.causal_in_time and i == train_rollout_limit - 1:
            y_preds.append(y_pred)
        else:
            y_preds.append(y_pred[:, -1:])

    y_pred_out = torch.cat(y_preds, dim=1)

    if mask is not None:
        mask_ref = expand_mask_to_match(mask, y_ref)
        y_ref.masked_fill_(mask_ref, 0)

    return y_pred_out, y_ref


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(
    y_pred: torch.Tensor,
    y_ref: torch.Tensor,
    field_names: List[str],
    padded_field_mask: torch.Tensor,
) -> Dict[str, Dict[str, float]]:
    """Compute per-field error metrics.

    Parameters
    ----------
    y_pred, y_ref : torch.Tensor
        Tensors of shape ``[B, T, H, W, D, C]`` (all channels, including padding).
    field_names : list[str]
        Names of *real* (non-padding) fields.
    padded_field_mask : torch.Tensor
        Boolean mask of shape ``[C]``.

    Returns
    -------
    dict
        ``{field_name: {metric_name: value}}`` plus an ``"__overall__"`` entry.
    """
    pred_real = y_pred[..., padded_field_mask].cpu().numpy()
    ref_real = y_ref[..., padded_field_mask].cpu().numpy()

    metrics: Dict[str, Dict[str, float]] = {}

    for i, name in enumerate(field_names):
        p = pred_real[0, :, :, :, :, i] if pred_real.ndim == 6 else pred_real[0, :, :, :, i]
        r = ref_real[0, :, :, :, :, i] if ref_real.ndim == 6 else ref_real[0, :, :, :, i]

        mse = float(np.mean((p - r) ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(p - r)))
        rel_l2 = float(np.linalg.norm(p - r) / max(np.linalg.norm(r), 1e-12))
        ss_res = float(np.sum((r - p) ** 2))
        ss_tot = float(np.sum((r - np.mean(r)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else float("nan")

        metrics[name] = {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "Rel_L2": rel_l2,
            "R2": r2,
        }

    # Overall averages
    metrics["__overall__"] = {
        k: float(np.mean([m[k] for m in metrics.values() if "__" not in k]))
        for k in ["MSE", "RMSE", "MAE", "Rel_L2", "R2"]
    }
    # Fix: recompute properly (the dict comprehension above includes __overall__ key check issue)
    field_metrics = [v for k, v in metrics.items() if k != "__overall__"]
    metrics["__overall__"] = {
        k: float(np.nanmean([m[k] for m in field_metrics]))
        for k in ["MSE", "RMSE", "MAE", "Rel_L2", "R2"]
    }

    return metrics


def print_metrics(metrics: Dict[str, Dict[str, float]]) -> None:
    """Pretty-print metrics to stdout."""
    print("\n" + "=" * 60)
    print("PERFORMANCE METRICS")
    print("=" * 60)

    for name, m in metrics.items():
        if name == "__overall__":
            continue
        print(f"\n{name.upper()}:")
        for k, v in m.items():
            print(f"  {k:12s}: {v:.6e}")

    overall = metrics["__overall__"]
    print(f"\nOVERALL (averaged across fields):")
    for k, v in overall.items():
        print(f"  {k:12s}: {v:.6e}")
    print("=" * 60)


# =============================================================================
# Plotting
# =============================================================================

def plot_spatial_comparison(
    y_pred: torch.Tensor,
    y_ref: torch.Tensor,
    field_names: List[str],
    padded_field_mask: torch.Tensor,
    T_in: int,
    output_path: str,
    timesteps_to_plot: Optional[List[int]] = None,
) -> None:
    """Spatial comparison: ground truth | prediction | absolute error."""
    pred_real = y_pred[..., padded_field_mask].cpu().numpy()
    ref_real = y_ref[..., padded_field_mask].cpu().numpy()
    T_out = pred_real.shape[1]

    if timesteps_to_plot is None:
        timesteps_to_plot = [0, T_out // 2, T_out - 1]
    timesteps_to_plot = [t for t in timesteps_to_plot if t < T_out]

    n_fields = len(field_names)
    n_times = len(timesteps_to_plot)

    fig = plt.figure(figsize=(6 * n_times, 4 * n_fields))
    gs = GridSpec(n_fields, n_times * 3, figure=fig, hspace=0.35, wspace=0.35)

    for fi, fname in enumerate(field_names):
        for ti, t in enumerate(timesteps_to_plot):
            # Squeeze out depth dim for 2D plotting
            p = np.squeeze(pred_real[0, t, ..., fi])
            r = np.squeeze(ref_real[0, t, ..., fi])
            err = np.abs(p - r)

            vmin = min(p.min(), r.min())
            vmax = max(p.max(), r.max())

            ax_r = fig.add_subplot(gs[fi, ti * 3])
            im = ax_r.imshow(r, cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
            ax_r.set_title(f"{fname}\nGT (t={t + T_in})", fontsize=9)
            ax_r.axis("off")
            plt.colorbar(im, ax=ax_r, fraction=0.046, pad=0.04)

            ax_p = fig.add_subplot(gs[fi, ti * 3 + 1])
            im = ax_p.imshow(p, cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
            ax_p.set_title(f"Pred (t={t + T_in})", fontsize=9)
            ax_p.axis("off")
            plt.colorbar(im, ax=ax_p, fraction=0.046, pad=0.04)

            ax_e = fig.add_subplot(gs[fi, ti * 3 + 2])
            im = ax_e.imshow(err, cmap="hot", origin="lower")
            ax_e.set_title(f"|Error| (t={t + T_in})", fontsize=9)
            ax_e.axis("off")
            plt.colorbar(im, ax=ax_e, fraction=0.046, pad=0.04)

    plt.suptitle("Walrus ZSL: Spatial Comparison", fontsize=14, y=1.0)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved spatial comparison  -> {output_path}")


def plot_temporal_errors(
    y_pred: torch.Tensor,
    y_ref: torch.Tensor,
    field_names: List[str],
    padded_field_mask: torch.Tensor,
    T_in: int,
    output_path: str,
) -> None:
    """MSE and MAE over time for each field."""
    pred_real = y_pred[..., padded_field_mask].cpu().numpy()
    ref_real = y_ref[..., padded_field_mask].cpu().numpy()
    T_out = pred_real.shape[1]
    n_fields = len(field_names)

    fig, axes = plt.subplots(1, n_fields, figsize=(6 * n_fields, 5))
    if n_fields == 1:
        axes = [axes]
    fig.suptitle("Temporal Evolution of Prediction Errors", fontsize=14)

    for fi, fname in enumerate(field_names):
        ax = axes[fi]
        mse_t, mae_t = [], []
        for t in range(T_out):
            p = np.squeeze(pred_real[0, t, ..., fi])
            r = np.squeeze(ref_real[0, t, ..., fi])
            mse_t.append(np.mean((p - r) ** 2))
            mae_t.append(np.mean(np.abs(p - r)))

        ts = np.arange(T_in, T_in + T_out)
        ax2 = ax.twinx()
        l1 = ax.plot(ts, mse_t, "b-o", label="MSE", lw=2, ms=4)
        l2 = ax2.plot(ts, mae_t, "r-s", label="MAE", lw=2, ms=4)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("MSE", color="b")
        ax.set_title(fname, fontweight="bold")
        ax.tick_params(axis="y", labelcolor="b")
        ax.grid(True, alpha=0.3)
        ax2.set_ylabel("MAE", color="r")
        ax2.tick_params(axis="y", labelcolor="r")
        lines = l1 + l2
        ax.legend(lines, [l.get_label() for l in lines], loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved temporal errors     -> {output_path}")


def plot_scatter(
    y_pred: torch.Tensor,
    y_ref: torch.Tensor,
    field_names: List[str],
    padded_field_mask: torch.Tensor,
    metrics: Dict[str, Dict[str, float]],
    output_path: str,
) -> None:
    """2D histogram scatter: ground truth vs prediction."""
    pred_real = y_pred[..., padded_field_mask].cpu().numpy()
    ref_real = y_ref[..., padded_field_mask].cpu().numpy()
    n_fields = len(field_names)

    fig, axes = plt.subplots(1, n_fields, figsize=(6 * n_fields, 5))
    if n_fields == 1:
        axes = [axes]
    fig.suptitle("Prediction vs Ground Truth Distribution", fontsize=14)

    for fi, fname in enumerate(field_names):
        ax = axes[fi]
        p = pred_real[..., fi].flatten()
        r = ref_real[..., fi].flatten()
        hist, xe, ye = np.histogram2d(r, p, bins=100)
        ax.imshow(
            hist.T, origin="lower", aspect="auto", cmap="viridis",
            extent=[xe[0], xe[-1], ye[0], ye[-1]],
        )
        lo = min(xe[0], ye[0])
        hi = max(xe[-1], ye[-1])
        ax.plot([lo, hi], [lo, hi], "r--", lw=2, label="Perfect")
        r2 = metrics[fname]["R2"]
        ax.set_title(f"{fname} (R²={r2:.4f})", fontsize=12)
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Prediction")
        ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved scatter plots       -> {output_path}")


# =============================================================================
# Main entry point
# =============================================================================

def run_zsl(
    extractor: Extractor,
    config: ZSLConfig,
    device: Optional[torch.device] = None,
    skip_plots: bool = False,
) -> ZSLResults:
    """Run the full zero-shot learning pipeline.

    Parameters
    ----------
    extractor : callable
        A function ``(ZSLConfig) -> ExtractedData`` that loads and formats
        your dataset.
    config : ZSLConfig
        Run configuration.
    device : torch.device or None
        Compute device. Auto-detected if ``None``.
    skip_plots : bool
        If ``True``, skip all visualisation.

    Returns
    -------
    ZSLResults
        Predictions, references, field names, and computed metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # 1. Extract data
    # ------------------------------------------------------------------
    print("\n[1/5] Extracting data ...")
    data = extractor(config)
    print(f"  Dataset:     {data.dataset_name}")
    print(f"  Fields:      {[s.name for s in data.field_specs]}")
    print(f"  Spatial:     {data.n_spatial_dims}D")
    print(f"  Time steps:  {data.fields.shape[0]}")
    print(f"  Resolution:  {data.fields.shape[1:-1]}")

    # ------------------------------------------------------------------
    # 2. Build batch
    # ------------------------------------------------------------------
    print("\n[2/5] Building Walrus batch ...")
    batch = _build_batch(data, config, device)
    print(f"  input_fields:  {batch['input_fields'].shape}")
    print(f"  output_fields: {batch['output_fields'].shape}")
    print(f"  field_indices: {batch['field_indices'].tolist()}")
    print(f"  padded_mask:   {batch['padded_field_mask'].tolist()}")
    print(f"  BCs:           {batch['boundary_conditions'].tolist()}")

    # ------------------------------------------------------------------
    # 3. Load model
    # ------------------------------------------------------------------
    print("\n[3/5] Loading pretrained Walrus model ...")
    model, revin, _ = _load_model(config, device)
    print(f"  Model loaded on {device}")

    # ------------------------------------------------------------------
    # 4. Rollout
    # ------------------------------------------------------------------
    print(f"\n[4/5] Running autoregressive rollout (max. up to {config.max_rollout_steps} steps) ...")
    formatter = ChannelsFirstWithTimeFormatter()
    with torch.no_grad():
        y_pred, y_ref = rollout(
            model, revin, batch, formatter,
            max_rollout_steps=config.max_rollout_steps,
            model_epsilon=config.model_epsilon,
            device=device,
        )
    print(f"  Predictions: {y_pred.shape}")
    print(f"  References:  {y_ref.shape}")

    # ------------------------------------------------------------------
    # 5. Evaluate
    # ------------------------------------------------------------------
    padded_mask = batch["padded_field_mask"].cpu()
    real_field_names = [s.name for s in data.field_specs if not s.is_padding]
    all_field_names = [s.name for s in data.field_specs]

    print("\n[5/5] Computing metrics ...")
    metrics = compute_metrics(y_pred, y_ref, real_field_names, padded_mask)
    print_metrics(metrics)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    if not skip_plots:
        out = Path(config.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        prefix = data.dataset_name.replace(" ", "_")

        print("\nGenerating plots ...")
        plot_spatial_comparison(
            y_pred, y_ref, real_field_names, padded_mask,
            config.T_in, str(out / f"{prefix}_spatial.png"),
        )
        plot_temporal_errors(
            y_pred, y_ref, real_field_names, padded_mask,
            config.T_in, str(out / f"{prefix}_temporal.png"),
        )
        plot_scatter(
            y_pred, y_ref, real_field_names, padded_mask,
            metrics, str(out / f"{prefix}_scatter.png"),
        )

    return ZSLResults(
        predictions=y_pred[..., padded_mask],
        references=y_ref[..., padded_mask],
        predictions_all=y_pred,
        references_all=y_ref,
        real_field_names=real_field_names,
        all_field_names=all_field_names,
        metrics=metrics,
        config=config,
    )
