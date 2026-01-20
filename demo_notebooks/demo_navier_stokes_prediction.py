# %%
"""
Walrus Navier-Stokes Prediction Demo
=====================================

This script demonstrates how to use a trained Walrus model to make predictions
on the Navier-Stokes dataset. It includes:
- Loading the converted Navier-Stokes dataset from Well format
- Initializing the Walrus IsotropicModel
- Running predictions on u (horizontal velocity), v (vertical velocity), and p (pressure)
- Computing Mean Squared Error (MSE) metrics
- Generating comprehensive validation plots

Author: Claude Code
Date: 2026-01-20
"""

# %%
# Imports
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

# Add walrus to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from walrus.data.multidatamodule import MixedWellDataModule
from walrus.models.isotropic_model import IsotropicModel

# %%
# Function: Load Navier-Stokes Data from Well Format
def load_navier_stokes_data(
    data_path: str,
    n_trajectories: int = 5,
    n_timesteps_input: int = 10,
    n_timesteps_predict: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Load Navier-Stokes data from Well-formatted HDF5 file.

    This function extracts u, v, and p fields from the converted Navier-Stokes
    dataset and prepares them for model input.

    Args:
        data_path: Path to the HDF5 file containing Navier-Stokes data
        n_trajectories: Number of trajectories to load
        n_timesteps_input: Number of input timesteps (context window)
        n_timesteps_predict: Number of timesteps to predict

    Returns:
        input_data: Tensor of shape [B, T_in, H, W, C] where C=3 (u, v, p)
        target_data: Tensor of shape [B, T_out, H, W, C] for validation
        metadata: Dictionary containing grid information and field names
    """
    print(f"Loading Navier-Stokes data from: {data_path}")

    with h5py.File(data_path, 'r') as f:
        # Extract dimensions
        n_traj_available = f.attrs['n_trajectories']
        n_traj = min(n_trajectories, n_traj_available)

        # Load velocity field [B, T, H, W, 2] where last dim is [u, v]
        velocity = f['t1_fields/velocity'][:n_traj, :, :, :, :]
        u = velocity[:, :, :, :, 0]  # Horizontal velocity [B, T, H, W]
        v = velocity[:, :, :, :, 1]  # Vertical velocity [B, T, H, W]

        # Load pressure field [B, T, H, W]
        p = f['t0_fields/pressure'][:n_traj, :, :, :]

        # Load grid information
        x_coords = f['dimensions/x'][:]
        y_coords = f['dimensions/y'][:]
        time = f['dimensions/time'][:]

        # Get boundary condition info
        bc_type = f['boundary_conditions/x_periodic'].attrs['bc_type']

        print(f"\nDataset Information:")
        print(f"  - Trajectories: {n_traj}")
        print(f"  - Total timesteps: {velocity.shape[1]}")
        print(f"  - Spatial resolution: {velocity.shape[2]} × {velocity.shape[3]}")
        print(f"  - Boundary conditions: {bc_type}")
        print(f"  - Field shapes:")
        print(f"      u: {u.shape}")
        print(f"      v: {v.shape}")
        print(f"      p: {p.shape}")

    # Stack fields: [B, T, H, W, C] where C=3 (u, v, p)
    # This is the format expected by Walrus
    data = np.stack([u, v, p], axis=-1)  # [B, T, H, W, 3]

    # Convert to torch tensors
    data = torch.from_numpy(data).float()

    # Split into input and target
    # Input: first n_timesteps_input frames
    # Target: next n_timesteps_predict frames
    total_needed = n_timesteps_input + n_timesteps_predict
    if data.shape[1] < total_needed:
        raise ValueError(
            f"Not enough timesteps in data. Need {total_needed}, have {data.shape[1]}"
        )

    input_data = data[:, :n_timesteps_input, :, :, :]  # [B, T_in, H, W, C]
    target_data = data[:, n_timesteps_input:n_timesteps_input+n_timesteps_predict, :, :, :]  # [B, T_out, H, W, C]

    # Prepare metadata dictionary
    metadata = {
        'field_names': ['u', 'v', 'p'],
        'x_coords': x_coords,
        'y_coords': y_coords,
        'time': time[:total_needed],
        'bc_type': bc_type,
        'n_spatial_dims': 2,
    }

    print(f"\nPrepared data for model:")
    print(f"  - Input shape: {input_data.shape}  # [B, T_in, H, W, C]")
    print(f"  - Target shape: {target_data.shape}  # [B, T_out, H, W, C]")

    return input_data, target_data, metadata


# %%
# Function: Initialize Walrus Model
def initialize_walrus_model(
    n_states: int = 3,
    hidden_dim: int = 768,
    projection_dim: int = 96,
    intermediate_dim: int = 192,
    processor_blocks: int = 12,
    checkpoint_path: str = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
) -> IsotropicModel:
    """
    Initialize the Walrus IsotropicModel for Navier-Stokes prediction.

    This function creates a Walrus model instance with the specified architecture
    parameters and optionally loads pretrained weights.

    Args:
        n_states: Number of input state variables (3 for u, v, p)
        hidden_dim: Hidden dimension of the model
        projection_dim: Projection dimension for embeddings
        intermediate_dim: Intermediate dimension in attention blocks
        processor_blocks: Number of space-time attention blocks
        checkpoint_path: Optional path to pretrained model checkpoint
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        model: Initialized IsotropicModel ready for inference
    """
    print(f"\nInitializing Walrus model...")
    print(f"  - Device: {device}")
    print(f"  - Number of states: {n_states}")
    print(f"  - Hidden dimension: {hidden_dim}")
    print(f"  - Processor blocks: {processor_blocks}")

    # Load configuration from Hydra configs
    config_path = str(Path(__file__).parent / "walrus" / "configs")

    try:
        with initialize_config_dir(version_base=None, config_dir=config_path):
            cfg = compose(config_name="config")

            # Override model parameters
            cfg.model.n_states = n_states
            cfg.model.hidden_dim = hidden_dim
            cfg.model.projection_dim = projection_dim
            cfg.model.intermediate_dim = intermediate_dim
            cfg.model.processor_blocks = processor_blocks

            # Instantiate model using Hydra config
            from hydra.utils import instantiate
            model = instantiate(cfg.model)

    except Exception as e:
        print(f"Warning: Could not load Hydra config: {e}")
        print("Falling back to manual initialization...")

        from walrus.models.encoders.vstride_encoder import VstrideEncoder, AdaptiveDVstrideEncoder, SpaceBagAdaptiveDVstrideEncoder
        from walrus.models.spatiotemporal_blocks.space_time_split import SpaceTimeSplitBlock

        encoder = SpaceBagAdaptiveDVstrideEncoder(
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
            max_d=3,
        )

        decoder = VStrideDecoder(
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
            max_d=3,
        )

        processor = SpaceTimeSplitProcessor(
            hidden_dim=hidden_dim,
            num_blocks=processor_blocks,
            intermediate_dim=intermediate_dim,
        )

        model = IsotropicModel(
            encoder=encoder,
            decoder=decoder,
            processor=processor,
            projection_dim=projection_dim,
            intermediate_dim=intermediate_dim,
            hidden_dim=hidden_dim,
            processor_blocks=processor_blocks,
            n_states=n_states,
            drop_path=0.05,
            groups=12,
            max_d=3,
            causal_in_time=False,
            jitter_patches=True,
            gradient_checkpointing_freq=0,
        )

    # Load checkpoint if provided
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"  - Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)

        print("  ✓ Checkpoint loaded successfully")
    else:
        print("  - No checkpoint provided, using randomly initialized weights")
        print("  - Note: For accurate predictions, load a pretrained checkpoint")

    # Move model to device and set to eval mode
    model = model.to(device)
    model.eval()

    print("  ✓ Model initialized successfully")

    return model


# %%
# Function: Prepare Model Input
def prepare_model_input(
    input_data: torch.Tensor,
    metadata: Dict,
    device: str = 'cuda',
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """
    Prepare input data in the format expected by Walrus model.

    The Walrus model expects input in the format:
    x: [T, B, C, H, W] (note the time-first dimension ordering)

    Args:
        input_data: Input tensor of shape [B, T, H, W, C]
        metadata: Metadata dictionary with grid info
        device: Device to move tensors to

    Returns:
        x: Input tensor in model format [T, B, C, H, W]
        state_labels: Channel labels [C]
        bcs: Boundary conditions [n_dims, 2]
        metadata_prepared: Prepared metadata for model
    """
    # Rearrange from [B, T, H, W, C] to [T, B, C, H, W]
    x = rearrange(input_data, 'b t h w c -> t b c h w')
    x = x.to(device)

    # State labels: simple integer indices for each channel
    n_channels = x.shape[2]
    state_labels = torch.arange(n_channels, device=device)

    # Boundary conditions: [n_dims, 2] where each row is [lower, upper] BC for that dimension
    # For periodic BCs, use code 2 (as per the Well format convention)
    # BC codes: 0=WALL, 1=OPEN, 2=PERIODIC
    n_spatial_dims = metadata['n_spatial_dims']
    bcs = torch.full((n_spatial_dims, 2), 2, dtype=torch.float32, device=device)  # PERIODIC

    # Prepare metadata in format expected by model
    metadata_prepared = {
        'field_names': metadata['field_names'],
        'grid': {
            'x': torch.from_numpy(metadata['x_coords']).to(device),
            'y': torch.from_numpy(metadata['y_coords']).to(device),
        },
        'n_spatial_dims': n_spatial_dims,
    }

    return x, state_labels, bcs, metadata_prepared


# %%
# Function: Run Autoregressive Prediction
def run_autoregressive_prediction(
    model: IsotropicModel,
    input_data: torch.Tensor,
    n_steps: int,
    metadata: Dict,
    device: str = 'cuda',
) -> torch.Tensor:
    """
    Run autoregressive prediction with the Walrus model.

    This function performs multi-step prediction by:
    1. Using the initial context window to predict the next timestep
    2. Appending the prediction to the context
    3. Sliding the context window forward
    4. Repeating for n_steps

    Args:
        model: Trained Walrus model
        input_data: Initial context [B, T_in, H, W, C]
        n_steps: Number of timesteps to predict
        metadata: Metadata dictionary
        device: Device for computation

    Returns:
        predictions: Predicted fields [B, n_steps, H, W, C]
    """
    print(f"\nRunning autoregressive prediction for {n_steps} steps...")

    model.eval()
    predictions = []

    # Start with the input context
    current_context = input_data.clone()  # [B, T_in, H, W, C]

    with torch.no_grad():
        for step in range(n_steps):
            # Prepare input for model
            x, state_labels, bcs, metadata_prepared = prepare_model_input(
                current_context, metadata, device
            )

            # Forward pass through model
            # Model returns: [T, B, C, H, W]
            output = model(
                x=x,
                state_labels=state_labels,
                bcs=bcs,
                metadata=metadata_prepared,
                train=False,
            )

            # Take the last timestep prediction: [B, C, H, W]
            pred_t = output[-1, :, :, :, :]  # Last time step

            # Rearrange to [B, H, W, C]
            pred_t = rearrange(pred_t, 'b c h w -> b h w c')

            # Store prediction
            predictions.append(pred_t.cpu())

            # Update context: remove oldest frame, append new prediction
            # [B, T_in, H, W, C] -> [B, T_in-1, H, W, C] + [B, 1, H, W, C]
            current_context = torch.cat([
                current_context[:, 1:, :, :, :],  # Remove first frame
                pred_t.unsqueeze(1)  # Add new prediction
            ], dim=1)

            if (step + 1) % 10 == 0 or step == n_steps - 1:
                print(f"  - Completed step {step + 1}/{n_steps}")

    # Stack predictions: [B, n_steps, H, W, C]
    predictions = torch.stack(predictions, dim=1)

    print(f"  ✓ Prediction complete. Output shape: {predictions.shape}")

    return predictions


# %%
# Function: Compute Metrics
def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    field_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Compute evaluation metrics (MSE, RMSE, MAE, R²) for predictions.

    Args:
        predictions: Predicted fields [B, T, H, W, C]
        targets: Ground truth fields [B, T, H, W, C]
        field_names: Names of fields (e.g., ['u', 'v', 'p'])

    Returns:
        metrics: Dictionary containing metrics for each field and timestep
    """
    print("\nComputing evaluation metrics...")

    B, T, H, W, C = predictions.shape

    # Initialize metrics storage
    metrics = {
        'per_field': {},
        'per_timestep': {},
        'overall': {},
    }

    # Compute metrics for each field
    for c, field_name in enumerate(field_names):
        pred_field = predictions[:, :, :, :, c]  # [B, T, H, W]
        target_field = targets[:, :, :, :, c]

        # Mean Squared Error (MSE)
        mse = torch.mean((pred_field - target_field) ** 2).item()

        # Root Mean Squared Error (RMSE)
        rmse = np.sqrt(mse)

        # Mean Absolute Error (MAE)
        mae = torch.mean(torch.abs(pred_field - target_field)).item()

        # R² Score (coefficient of determination)
        ss_res = torch.sum((target_field - pred_field) ** 2).item()
        ss_tot = torch.sum((target_field - torch.mean(target_field)) ** 2).item()
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Normalized RMSE (by standard deviation)
        std = torch.std(target_field).item()
        nrmse = rmse / std if std > 0 else 0.0

        metrics['per_field'][field_name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'NRMSE': nrmse,
        }

        print(f"\n  {field_name}:")
        print(f"    - MSE:   {mse:.6e}")
        print(f"    - RMSE:  {rmse:.6e}")
        print(f"    - MAE:   {mae:.6e}")
        print(f"    - R²:    {r2:.4f}")
        print(f"    - NRMSE: {nrmse:.4f}")

    # Compute metrics per timestep (averaged over all fields)
    for t in range(T):
        pred_t = predictions[:, t, :, :, :]  # [B, H, W, C]
        target_t = targets[:, t, :, :, :]

        mse_t = torch.mean((pred_t - target_t) ** 2).item()
        rmse_t = np.sqrt(mse_t)

        metrics['per_timestep'][f't{t}'] = {
            'MSE': mse_t,
            'RMSE': rmse_t,
        }

    # Overall metrics (all fields, all timesteps)
    overall_mse = torch.mean((predictions - targets) ** 2).item()
    overall_rmse = np.sqrt(overall_mse)
    overall_mae = torch.mean(torch.abs(predictions - targets)).item()

    metrics['overall'] = {
        'MSE': overall_mse,
        'RMSE': overall_rmse,
        'MAE': overall_mae,
    }

    print(f"\n  Overall (all fields, all timesteps):")
    print(f"    - MSE:  {overall_mse:.6e}")
    print(f"    - RMSE: {overall_rmse:.6e}")
    print(f"    - MAE:  {overall_mae:.6e}")

    return metrics


# %%
# Function: Create Validation Plots
def create_validation_plots(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    input_data: torch.Tensor,
    metadata: Dict,
    metrics: Dict,
    output_dir: str = './output',
    trajectory_idx: int = 0,
) -> None:
    """
    Create comprehensive validation plots for Navier-Stokes predictions.

    This function generates:
    1. Field snapshots at different timesteps
    2. Difference (error) plots between prediction and ground truth
    3. Temporal evolution plots for specific spatial points
    4. MSE evolution over time
    5. Spatial error distribution

    Args:
        predictions: Predicted fields [B, T, H, W, C]
        targets: Ground truth fields [B, T, H, W, C]
        input_data: Input context [B, T_in, H, W, C]
        metadata: Metadata dictionary
        metrics: Computed metrics dictionary
        output_dir: Directory to save plots
        trajectory_idx: Which trajectory to visualize
    """
    print(f"\nGenerating validation plots...")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    field_names = metadata['field_names']
    B, T, H, W, C = predictions.shape

    # Convert to numpy for plotting
    preds_np = predictions[trajectory_idx].numpy()  # [T, H, W, C]
    targets_np = targets[trajectory_idx].numpy()
    input_np = input_data[trajectory_idx].numpy()  # [T_in, H, W, C]

    # ------------------------------
    # Plot 1: Field Snapshots at Multiple Timesteps
    # ------------------------------
    print("  - Creating field snapshot comparison plots...")

    timesteps_to_plot = [0, T//2, T-1]  # First, middle, last

    for t_idx in timesteps_to_plot:
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle(f'Field Comparison at t={t_idx} (Trajectory {trajectory_idx})',
                     fontsize=16, fontweight='bold')

        for c, field_name in enumerate(field_names):
            # Ground truth
            im0 = axes[c, 0].imshow(
                targets_np[t_idx, :, :, c],
                cmap='RdBu_r' if field_name in ['u', 'v'] else 'plasma',
                origin='lower',
                aspect='auto',
            )
            axes[c, 0].set_title(f'{field_name} - Ground Truth')
            axes[c, 0].set_ylabel('y')
            plt.colorbar(im0, ax=axes[c, 0])

            # Prediction
            im1 = axes[c, 1].imshow(
                preds_np[t_idx, :, :, c],
                cmap='RdBu_r' if field_name in ['u', 'v'] else 'plasma',
                origin='lower',
                aspect='auto',
            )
            axes[c, 1].set_title(f'{field_name} - Prediction')
            plt.colorbar(im1, ax=axes[c, 1])

            # Absolute difference
            diff = np.abs(preds_np[t_idx, :, :, c] - targets_np[t_idx, :, :, c])
            im2 = axes[c, 2].imshow(
                diff,
                cmap='hot',
                origin='lower',
                aspect='auto',
            )
            axes[c, 2].set_title(f'{field_name} - |Error|')
            plt.colorbar(im2, ax=axes[c, 2])

            # Relative error (%)
            # Avoid division by zero
            target_range = np.max(np.abs(targets_np[t_idx, :, :, c]))
            if target_range > 1e-10:
                rel_error = 100 * diff / target_range
            else:
                rel_error = np.zeros_like(diff)

            im3 = axes[c, 3].imshow(
                rel_error,
                cmap='hot',
                origin='lower',
                aspect='auto',
                vmin=0,
                vmax=10,  # Cap at 10% for better visualization
            )
            axes[c, 3].set_title(f'{field_name} - Relative Error (%)')
            plt.colorbar(im3, ax=axes[c, 3])

        # Add x-labels to bottom row
        for ax in axes[-1, :]:
            ax.set_xlabel('x')

        plt.tight_layout()
        plt.savefig(output_path / f'field_comparison_t{t_idx}.png', dpi=150, bbox_inches='tight')
        plt.close()

    print(f"    ✓ Saved field comparison plots for {len(timesteps_to_plot)} timesteps")

    # ------------------------------
    # Plot 2: Temporal Evolution at Specific Spatial Points
    # ------------------------------
    print("  - Creating temporal evolution plots...")

    # Select a few spatial points to track over time
    spatial_points = [
        (H//4, W//4),      # Bottom-left quadrant
        (H//2, W//2),      # Center
        (3*H//4, 3*W//4),  # Top-right quadrant
    ]

    fig, axes = plt.subplots(len(field_names), 1, figsize=(12, 4*len(field_names)))
    if len(field_names) == 1:
        axes = [axes]

    fig.suptitle(f'Temporal Evolution at Spatial Points (Trajectory {trajectory_idx})',
                 fontsize=14, fontweight='bold')

    for c, field_name in enumerate(field_names):
        for h, w in spatial_points:
            # Plot ground truth
            axes[c].plot(
                range(T),
                targets_np[:, h, w, c],
                '--',
                label=f'GT at ({h}, {w})',
                alpha=0.7,
            )

            # Plot prediction
            axes[c].plot(
                range(T),
                preds_np[:, h, w, c],
                '-',
                label=f'Pred at ({h}, {w})',
            )

        axes[c].set_ylabel(field_name)
        axes[c].set_xlabel('Timestep')
        axes[c].legend(loc='best', fontsize=8)
        axes[c].grid(True, alpha=0.3)
        axes[c].set_title(f'{field_name} Evolution')

    plt.tight_layout()
    plt.savefig(output_path / 'temporal_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"    ✓ Saved temporal evolution plot")

    # ------------------------------
    # Plot 3: MSE Evolution Over Time
    # ------------------------------
    print("  - Creating MSE evolution plot...")

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for c, field_name in enumerate(field_names):
        mse_per_t = []
        for t in range(T):
            mse_t = np.mean((preds_np[t, :, :, c] - targets_np[t, :, :, c]) ** 2)
            mse_per_t.append(mse_t)

        ax.plot(range(T), mse_per_t, marker='o', label=f'{field_name} MSE', linewidth=2)

    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Mean Squared Error', fontsize=12)
    ax.set_title('MSE Evolution Over Prediction Horizon', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale for better visualization

    plt.tight_layout()
    plt.savefig(output_path / 'mse_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"    ✓ Saved MSE evolution plot")

    # ------------------------------
    # Plot 4: Spatial Error Distribution
    # ------------------------------
    print("  - Creating spatial error distribution plots...")

    fig, axes = plt.subplots(1, len(field_names), figsize=(6*len(field_names), 5))
    if len(field_names) == 1:
        axes = [axes]

    fig.suptitle(f'Time-Averaged Spatial Error Distribution (Trajectory {trajectory_idx})',
                 fontsize=14, fontweight='bold')

    for c, field_name in enumerate(field_names):
        # Compute time-averaged absolute error
        error_avg = np.mean(np.abs(preds_np[:, :, :, c] - targets_np[:, :, :, c]), axis=0)

        im = axes[c].imshow(
            error_avg,
            cmap='hot',
            origin='lower',
            aspect='auto',
        )
        axes[c].set_title(f'{field_name} - Avg |Error|')
        axes[c].set_xlabel('x')
        axes[c].set_ylabel('y')
        plt.colorbar(im, ax=axes[c])

    plt.tight_layout()
    plt.savefig(output_path / 'spatial_error_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"    ✓ Saved spatial error distribution plot")

    # ------------------------------
    # Plot 5: Error Histograms
    # ------------------------------
    print("  - Creating error histogram plots...")

    fig, axes = plt.subplots(1, len(field_names), figsize=(6*len(field_names), 4))
    if len(field_names) == 1:
        axes = [axes]

    fig.suptitle(f'Error Distribution Histograms (Trajectory {trajectory_idx})',
                 fontsize=14, fontweight='bold')

    for c, field_name in enumerate(field_names):
        # Flatten all errors for this field
        errors = (preds_np[:, :, :, c] - targets_np[:, :, :, c]).flatten()

        axes[c].hist(errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[c].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
        axes[c].set_xlabel('Prediction Error')
        axes[c].set_ylabel('Frequency')
        axes[c].set_title(f'{field_name} Error Distribution')
        axes[c].legend()
        axes[c].grid(True, alpha=0.3)

        # Add statistics text
        mean_err = np.mean(errors)
        std_err = np.std(errors)
        axes[c].text(
            0.05, 0.95,
            f'Mean: {mean_err:.2e}\nStd: {std_err:.2e}',
            transform=axes[c].transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path / 'error_histograms.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"    ✓ Saved error histogram plot")

    # ------------------------------
    # Plot 6: Metrics Summary
    # ------------------------------
    print("  - Creating metrics summary plot...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Extract metrics for plotting
    field_metrics = metrics['per_field']
    metric_types = ['MSE', 'RMSE', 'MAE', 'R2']

    for idx, metric_type in enumerate(metric_types):
        ax = axes[idx // 2, idx % 2]

        values = [field_metrics[field][metric_type] for field in field_names]

        bars = ax.bar(field_names, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(field_names)])
        ax.set_ylabel(metric_type, fontsize=12)
        ax.set_title(f'{metric_type} per Field', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{height:.2e}' if metric_type != 'R2' else f'{height:.4f}',
                ha='center',
                va='bottom',
                fontsize=9,
            )

    fig.suptitle(f'Evaluation Metrics Summary (Trajectory {trajectory_idx})',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'metrics_summary.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"    ✓ Saved metrics summary plot")

    print(f"\n  ✓ All plots saved to: {output_path}")


# %%
# Function: Save Metrics to File
def save_metrics_to_file(
    metrics: Dict,
    output_dir: str = './output',
    filename: str = 'metrics.txt',
) -> None:
    """
    Save computed metrics to a text file for easy reference.

    Args:
        metrics: Dictionary of computed metrics
        output_dir: Directory to save the metrics file
        filename: Name of the output file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filepath = output_path / filename

    with open(filepath, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("WALRUS NAVIER-STOKES PREDICTION METRICS\n")
        f.write("=" * 70 + "\n\n")

        # Per-field metrics
        f.write("METRICS PER FIELD:\n")
        f.write("-" * 70 + "\n")
        for field_name, field_metrics in metrics['per_field'].items():
            f.write(f"\n{field_name}:\n")
            for metric_name, value in field_metrics.items():
                f.write(f"  {metric_name:8s}: {value:.6e}\n")

        # Overall metrics
        f.write("\n" + "=" * 70 + "\n")
        f.write("OVERALL METRICS (all fields, all timesteps):\n")
        f.write("-" * 70 + "\n")
        for metric_name, value in metrics['overall'].items():
            f.write(f"  {metric_name:8s}: {value:.6e}\n")

        f.write("\n" + "=" * 70 + "\n")

    print(f"\n  ✓ Metrics saved to: {filepath}")


# %%
# Configuration
# ------------------------------
# Set your configuration parameters here
# ------------------------------

# Data configuration
DATA_PATH = "./converted_data/navier_stokes_spectral_id_n5.hdf5"
N_TRAJECTORIES = 5
N_TIMESTEPS_INPUT = 10  # Context window size
N_TIMESTEPS_PREDICT = 10  # Number of timesteps to predict

# Model configuration
N_STATES = 3  # u, v, p
HIDDEN_DIM = 768
PROJECTION_DIM = 96
INTERMEDIATE_DIM = 192
PROCESSOR_BLOCKS = 12
CHECKPOINT_PATH = None  # Set to path of pretrained checkpoint if available

# Output configuration
OUTPUT_DIR = "./output/navier_stokes_demo"

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# %%
# Print Configuration
print("=" * 80)
print("WALRUS NAVIER-STOKES PREDICTION DEMO")
print("=" * 80)
print(f"\nConfiguration:")
print(f"  - Data path: {DATA_PATH}")
print(f"  - Context window: {N_TIMESTEPS_INPUT} timesteps")
print(f"  - Prediction horizon: {N_TIMESTEPS_PREDICT} timesteps")
print(f"  - Fields: u, v, p")
print(f"  - Device: {DEVICE}")
print(f"  - Output directory: {OUTPUT_DIR}")

# %%
# Step 1: Load Data
input_data, target_data, metadata = load_navier_stokes_data(
    data_path=DATA_PATH,
    n_trajectories=N_TRAJECTORIES,
    n_timesteps_input=N_TIMESTEPS_INPUT,
    n_timesteps_predict=N_TIMESTEPS_PREDICT,
)

# %%
# Step 2: Initialize Model
model = initialize_walrus_model(
    n_states=N_STATES,
    hidden_dim=HIDDEN_DIM,
    projection_dim=PROJECTION_DIM,
    intermediate_dim=INTERMEDIATE_DIM,
    processor_blocks=PROCESSOR_BLOCKS,
    checkpoint_path=CHECKPOINT_PATH,
    device=DEVICE,
)

# %%
# Step 3: Run Predictions
predictions = run_autoregressive_prediction(
    model=model,
    input_data=input_data,
    n_steps=N_TIMESTEPS_PREDICT,
    metadata=metadata,
    device=DEVICE,
)

# %%
# Step 4: Compute Metrics
metrics = compute_metrics(
    predictions=predictions,
    targets=target_data,
    field_names=metadata['field_names'],
)

# %%
# Step 5: Generate Validation Plots
create_validation_plots(
    predictions=predictions,
    targets=target_data,
    input_data=input_data,
    metadata=metadata,
    metrics=metrics,
    output_dir=OUTPUT_DIR,
    trajectory_idx=0,  # Visualize first trajectory
)

# %%
# Step 6: Save Results
save_metrics_to_file(
    metrics=metrics,
    output_dir=OUTPUT_DIR,
    filename='metrics.txt',
)

# Save predictions and targets for further analysis
output_path = Path(OUTPUT_DIR)
torch.save({
    'predictions': predictions,
    'targets': target_data,
    'input': input_data,
    'metadata': metadata,
    'metrics': metrics,
}, output_path / 'results.pt')

print(f"\n  ✓ Results saved to: {output_path / 'results.pt'}")

# %%
# Summary
print("\n" + "=" * 80)
print("DEMO COMPLETE")
print("=" * 80)
print(f"\nSummary:")
print(f"  - Predicted {N_TIMESTEPS_PREDICT} timesteps for {N_TRAJECTORIES} trajectories")
print(f"  - Overall MSE: {metrics['overall']['MSE']:.6e}")
print(f"  - Overall RMSE: {metrics['overall']['RMSE']:.6e}")
print(f"  - Overall MAE: {metrics['overall']['MAE']:.6e}")
print(f"\nOutputs:")
print(f"  - Validation plots: {OUTPUT_DIR}/")
print(f"  - Metrics file: {OUTPUT_DIR}/metrics.txt")
print(f"  - Results file: {OUTPUT_DIR}/results.pt")
print("\n" + "=" * 80 + "\n")

# %%
