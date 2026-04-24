# %%
import torch
import torch.nn.functional as F
import h5py
import copy
from walrus.models import IsotropicModel
from walrus.data.well_to_multi_transformer import ChannelsFirstWithTimeFormatter
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from walrus.utils.experiment_utils import align_checkpoint_with_field_to_index_map
from the_well.data.datasets import WellMetadata
from the_well.data.utils import flatten_field_names
from walrus.trainer.training import expand_mask_to_match

# %% 
# =============================================================================
# STEP 1: Load your converted Navier-Stokes HDF5 file
# =============================================================================

hdf5_path = "/Users/Vicky/Documents/UKAEA/Code/Foundation_Models/walrus/demo_notebooks/converted_data/navier_stokes_spectral_id_n5.hdf5"

with h5py.File(hdf5_path, 'r') as f:
    # Load dimensions
    Nx, Ny = f['dimensions/x'].shape[0], f['dimensions/y'].shape[0]
    
    # Load one trajectory for this example (you can loop over all trajectories)
    traj_idx = 0
    
    # Load fields: velocity [Nt, Nx, Ny, 2], pressure [Nt, Nx, Ny]
    # Note: density is not loaded as we're only modeling u, v, and p
    velocity = torch.tensor(f['t1_fields/velocity'][traj_idx], dtype=torch.float32)  # [Nt, Nx, Ny, 2]
    pressure = torch.tensor(f['t0_fields/pressure'][traj_idx], dtype=torch.float32)  # [Nt, Nx, Ny]
    
    # Boundary condition type
    bc_type_map = {"WALL": 0, "OPEN": 1, "PERIODIC": 2}
    bc_x = f['boundary_conditions/x_periodic'].attrs['bc_type']
    bc_y = f['boundary_conditions/y_periodic'].attrs['bc_type']
    bc_code = bc_type_map[bc_x]  # Assuming same for both x and y
    
    print(f"Loaded Navier-Stokes data:")
    print(f"  Velocity shape: {velocity.shape}")
    print(f"  Pressure shape: {pressure.shape}")
    print(f"  Boundary conditions: {bc_x}, {bc_y}")

# %% # =============================================================================
# STEP 2: Prepare data in Walrus format (CORRECTED)
# =============================================================================

# Split into input and output timesteps
T_in = 4   # Number of input timesteps
T_out = 20  # Number of output timesteps to predict

# Extract velocity components
u = velocity[..., 0]  # [Nt, Nx, Ny]
v = velocity[..., 1]  # [Nt, Nx, Ny]

# Resize spatial dimensions to be compatible with model
# Model requires size // 32 to be in {0, 1, 4, 8, 12, 16, 24, 32}
# Valid sizes: 32, 128, 256, 384, 512, 768, 1024
target_size = 128
print(f"\nResizing from {Nx}x{Ny} to {target_size}x{target_size} (model requirement)")

# Resize u, v, pressure using bilinear interpolation
# Input shape for F.interpolate: [Nt, 1, Nx, Ny]
u = F.interpolate(u.unsqueeze(1), size=(target_size, target_size), mode='bilinear', align_corners=False).squeeze(1)
v = F.interpolate(v.unsqueeze(1), size=(target_size, target_size), mode='bilinear', align_corners=False).squeeze(1)
pressure = F.interpolate(pressure.unsqueeze(1), size=(target_size, target_size), mode='bilinear', align_corners=False).squeeze(1)

# Update Nx, Ny
Nx, Ny = target_size, target_size
print(f"Resized shapes: u={u.shape}, v={v.shape}, pressure={pressure.shape}")

# For 2D simulations, we need to add velocity_z as padding (zero)
# Stack: [velocity_x, velocity_y, velocity_z, pressure]

# Create zero tensor for velocity_z padding
velocity_z = torch.zeros_like(u)

# Stack all fields: [Nt, Nx, Ny, 4]
all_fields = torch.stack([u, v, velocity_z, pressure], dim=-1)

# Add depth dimension (D=1) for tensor format consistency
all_fields = all_fields.unsqueeze(-2)  # [Nt, Nx, Ny, 1, 4]

# Split into input and output
Nt_total = all_fields.shape[0]
input_fields = all_fields[:T_in].unsqueeze(0)   # [1, T_in, Nx, Ny, 1, 4]
output_fields = all_fields[T_in:T_in+T_out].unsqueeze(0)  # [1, T_out, Nx, Ny, 1, 4]

print(f"\nPrepared fields:")
print(f"  Input shape: {input_fields.shape}  # [B=1, T_in=6, H={Nx}, W={Ny}, D=1, C=4]")
print(f"  Output shape: {output_fields.shape}  # [B=1, T_out={T_out}, H={Nx}, W={Ny}, D=1, C=4]")

# %%
# =============================================================================
# STEP 3: Create field index mapping
# =============================================================================

# Map fields to pretrained Walrus embeddings:
# - velocity_x → index 4 (pretrained)
# - velocity_y → index 5 (pretrained)
# - velocity_z → index 6 (pretrained, used as padding)
# - pressure → index 3 (pretrained)

field_indices = torch.tensor([4, 5, 6, 3])  # [velocity_x, velocity_y, velocity_z, pressure]

# Padded field mask: True for real fields, False for padding
# velocity_z is padding (False), all others are real (True)
padded_field_mask = torch.tensor([True, True, False, True])

print(f"\nField mapping:")
field_names_list = ['velocity_x', 'velocity_y', 'velocity_z (padding)', 'pressure']
for i, (idx, name, is_real) in enumerate(zip(field_indices, field_names_list, padded_field_mask)):
    status = "real" if is_real else "padding"
    print(f"  Channel {i}: {name:30s} → embedding {idx:2d} ({status})")

# %%
# =============================================================================
# STEP 4: Create metadata
# =============================================================================

metadata = WellMetadata(
    dataset_name="navier_stokes_spectral",
    n_spatial_dims=3,  # Walrus expects 3D (we padded D=1)

    # Field organization by rank:
    # Rank 0 (scalars): pressure
    # Rank 1 (vectors): velocity_x, velocity_y, velocity_z
    field_names={
        0: ['pressure'] ,
        1: ['velocity_x', 'velocity_y', 'velocity_z'],
        2: []
    },
    
    spatial_resolution=(Nx, Ny, 1),
    scalar_names=[],
    constant_field_names={0: [], 1: [], 2: []},
    constant_scalar_names=[],
    boundary_condition_types=[],
    n_files=[],
    n_trajectories_per_file=[],
    n_steps_per_trajectory=[],
    grid_type='cartesian'
)
# %% # =============================================================================
# STEP 5: Create the batch dictionary (Walrus format)
# =============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

navier_stokes_batch = {
    # Input fields: [B, T_in, H, W, D, C]
    "input_fields": input_fields.to(device),
    
    # Output fields: [B, T_out, H, W, D, C]
    "output_fields": output_fields.to(device),
    
    # Constant fields: [B, H, W, D, C_const] - none in this case
    "constant_fields": torch.empty(1, Nx, Ny, 1, 0, device=device),
    
    # Boundary conditions: [B, 3, 2]
    # bc_code is 2 for PERIODIC
    "boundary_conditions": torch.tensor([[[bc_code, bc_code], 
                                          [bc_code, bc_code], 
                                          [bc_code, bc_code]]], device=device),
    
    # Padded field mask: [C]
    "padded_field_mask": padded_field_mask.to(device),
    
    # Field indices: [C]
    "field_indices": field_indices.to(device),
    
    # Metadata
    "metadata": metadata,
}

print(f"\n✓ Created Walrus batch dictionary")
print(f"  Device: {device}")
print(f"  Ready for model inference!")

# %%
# =============================================================================
# STEP 6: Load Walrus model (same as synthetic data example)
# =============================================================================

# Set paths to your downloaded model files
checkpoint_base_path = "/Users/Vicky/Documents/UKAEA/Code/Foundation_Models/walrus/demo_notebooks/checkpoints"
config_base_path = "/Users/Vicky/Documents/UKAEA/Code/Foundation_Models/walrus/demo_notebooks/configs"

checkpoint_path = f"{checkpoint_base_path}/walrus.pt"
checkpoint_config_path = f"{config_base_path}/extended_config.yaml"

# Load checkpoint and config
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["app"]["model"]
config = OmegaConf.load(checkpoint_config_path)

# Get field mapping and ensure our fields are included
field_to_index_map = config.data.field_index_map_override
new_field_to_index_map = dict(field_to_index_map)

# All our fields should already be in the pretrained model, but verify:
required_indices = [4, 5, 6, 3]  # velocity_x, velocity_y, velocity_z, pressure
print(f"\nVerifying field indices are in pretrained model:")
for idx in required_indices:
    field_name = [k for k, v in field_to_index_map.items() if v == idx]
    print(f"  Index {idx}: {field_name}")

# Initialize model
model = instantiate(
    config.model,
    n_states=max(new_field_to_index_map.values()) + 1,
)

# Load weights
revised_model_checkpoint = align_checkpoint_with_field_to_index_map(
    checkpoint_state_dict=checkpoint,
    model_state_dict=model.state_dict(),
    checkpoint_field_to_index_map=field_to_index_map,
    model_field_to_index_map=new_field_to_index_map,
)

model.load_state_dict(revised_model_checkpoint)
model.to(device)
model.eval()

print(f"\n✓ Model loaded successfully")

# %% 
# =============================================================================
# STEP 7: Setup helper objects
# =============================================================================

formatter = ChannelsFirstWithTimeFormatter()
revin = instantiate(config.trainer.revin)()

print(f"✓ Helper objects initialized")

# =============================================================================
# STEP 8: Define rollout function (same as synthetic example)
# =============================================================================

def rollout_model(model, revin, batch, formatter, max_rollout_steps=200, model_epsilon=1e-5, device=torch.device("cpu")):
    """Rollout the model autoregressively for multiple timesteps."""
    metadata = batch["metadata"]
    
    batch = {
        k: v.to(device)
        if k not in {"metadata", "boundary_conditions"}
        else v
        for k, v in batch.items()
    }
    
    # Check for mask
    if "mask" in batch["metadata"].constant_field_names[0]:
        mask_index = batch["metadata"].constant_field_names[0].index("mask")
        mask = batch["constant_fields"][..., mask_index : mask_index + 1]
        mask = mask.to(device, dtype=torch.bool)
    else:
        mask = None

    inputs, y_ref = formatter.process_input(
        batch,
        causal_in_time=model.causal_in_time,
        predict_delta=True,
        train=False,
    )

    T_in = batch["input_fields"].shape[1]
    max_rollout_steps = max_rollout_steps + (T_in - 1)
    rollout_steps = min(y_ref.shape[1], max_rollout_steps)
    train_rollout_limit = 1

    y_ref = y_ref[:, :rollout_steps]
    moving_batch = copy.deepcopy(batch)
    y_preds = []
    
    for i in range(train_rollout_limit - 1, rollout_steps):
        inputs, _ = formatter.process_input(moving_batch)
        inputs = list(inputs)
        
        with torch.no_grad():
            normalization_stats = revin.compute_stats(
                inputs[0],
                metadata,
                epsilon=model_epsilon
            )
        
        normalized_inputs = inputs[:]
        normalized_inputs[0] = revin.normalize_stdmean(
            normalized_inputs[0], 
            normalization_stats
        )
        
        y_pred = model(
            normalized_inputs[0],
            normalized_inputs[1],
            normalized_inputs[2].tolist(),
            metadata=metadata,
        )
        
        if model.causal_in_time:
            y_pred = y_pred[-1:]
        
        y_pred = (inputs[0][-y_pred.shape[0]:].float()
                  + revin.denormalize_delta(y_pred, normalization_stats))
        
        y_pred = formatter.process_output(y_pred, metadata)[..., : y_ref.shape[-1]]

        if mask is not None:
            mask_pred = expand_mask_to_match(mask, y_pred)
            y_pred.masked_fill_(mask_pred, 0)

        y_pred = y_pred.masked_fill(~batch["padded_field_mask"], 0.0)

        if i != rollout_steps - 1:
            moving_batch["input_fields"] = torch.cat(
                [moving_batch["input_fields"][:, 1:],
                 y_pred[:, -1:]],
                dim=1
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

# %% 
# =============================================================================
# STEP 9: Run inference on Navier-Stokes data
# =============================================================================

with torch.no_grad():
    print(f"\n" + "="*60)
    print("Running Walrus inference on Navier-Stokes data")
    print("="*60)
    
    y_pred, y_ref = rollout_model(
        model,
        revin,
        navier_stokes_batch,
        formatter,
        max_rollout_steps=20,
        device=device,
    )
    
    print(f"\n✓ Prediction complete!")
    print(f"  Prediction shape (with padding): {y_pred.shape}")
    print(f"  Reference shape (with padding): {y_ref.shape}")

    # Remove padding (velocity_z is at index 2)
    y_pred_real = y_pred[..., padded_field_mask]
    y_ref_real = y_ref[..., padded_field_mask]

    real_field_names = ['velocity_x', 'velocity_y', 'pressure']

    print(f"\n  Final shape (real fields only): {y_pred_real.shape}")
    print(f"  Fields: {real_field_names}")

# %%
# =============================================================================
# STEP 10: Performance Metrics
# =============================================================================

import numpy as np

print(f"\n" + "="*60)
print("PERFORMANCE METRICS")
print("="*60)

# Compute metrics for each field
metrics_per_field = {}

for i, field_name in enumerate(real_field_names):
    pred = y_pred_real[0, :, :, :, 0, i].cpu().numpy()
    ref = y_ref_real[0, :, :, :, 0, i].cpu().numpy()

    # Mean Squared Error (MSE)
    mse = np.mean((pred - ref) ** 2)

    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)

    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(pred - ref))

    # Relative L2 Error
    rel_l2 = np.linalg.norm(pred - ref) / np.linalg.norm(ref)

    # R-squared (coefficient of determination)
    ss_res = np.sum((ref - pred) ** 2)
    ss_tot = np.sum((ref - np.mean(ref)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else float('nan')

    metrics_per_field[field_name] = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'Rel_L2': rel_l2,
        'R2': r2
    }

    print(f"\n{field_name.upper()}:")
    print(f"  MSE:          {mse:.6e}")
    print(f"  RMSE:         {rmse:.6e}")
    print(f"  MAE:          {mae:.6e}")
    print(f"  Rel L2 Error: {rel_l2:.6f}")
    print(f"  R² Score:     {r2:.6f}")

# Overall metrics
overall_mse = np.mean([m['MSE'] for m in metrics_per_field.values()])
overall_mae = np.mean([m['MAE'] for m in metrics_per_field.values()])
overall_r2 = np.mean([m['R2'] for m in metrics_per_field.values()])

print(f"\nOVERALL METRICS (averaged across fields):")
print(f"  MSE:      {overall_mse:.6e}")
print(f"  MAE:      {overall_mae:.6e}")
print(f"  R² Score: {overall_r2:.6f}")

# Print data statistics for context
print(f"\nDATA STATISTICS (for context):")
for i, field_name in enumerate(real_field_names):
    pred = y_pred_real[0, :, :, :, 0, i].cpu().numpy()
    ref = y_ref_real[0, :, :, :, 0, i].cpu().numpy()
    print(f"\n{field_name}:")
    print(f"  Ground Truth - mean: {ref.mean():.6e}, std: {ref.std():.6e}, range: [{ref.min():.6e}, {ref.max():.6e}]")
    print(f"  Prediction   - mean: {pred.mean():.6e}, std: {pred.std():.6e}, range: [{pred.min():.6e}, {pred.max():.6e}]")

# Check if this is zero-shot learning scenario
if np.isnan(overall_r2) or overall_r2 < 0.5:
    print(f"\n⚠ NOTE: This is a zero-shot learning (ZSL) scenario.")
    print(f"  The model was NOT trained on Navier-Stokes data.")
    print(f"  The large errors suggest the model needs:")
    print(f"    - Proper normalization of the input data")
    print(f"    - Fine-tuning on similar fluid dynamics data")
    print(f"    - Or use of a model trained on related PDEs")

# %%
# =============================================================================
# STEP 11: Visualization and Comparison Plots
# =============================================================================

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Select timesteps to visualize
timesteps_to_plot = [0, 3, 6]  # Beginning, middle, end

# Create a comprehensive comparison plot
fig = plt.figure(figsize=(20, 12))
gs = GridSpec(len(real_field_names), len(timesteps_to_plot) * 3, figure=fig, hspace=0.3, wspace=0.3)

for field_idx, field_name in enumerate(real_field_names):
    for t_idx, t in enumerate(timesteps_to_plot):
        # Get data
        pred = y_pred_real[0, t, :, :, 0, field_idx].cpu().numpy()
        ref = y_ref_real[0, t, :, :, 0, field_idx].cpu().numpy()
        error = np.abs(pred - ref)

        # Determine shared colorbar range for pred and ref
        vmin = min(pred.min(), ref.min())
        vmax = max(pred.max(), ref.max())

        # Reference (Ground Truth)
        ax_ref = fig.add_subplot(gs[field_idx, t_idx * 3])
        im_ref = ax_ref.imshow(ref, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
        ax_ref.set_title(f'{field_name}\nGround Truth (t={t+T_in})', fontsize=10)
        ax_ref.axis('off')
        plt.colorbar(im_ref, ax=ax_ref, fraction=0.046, pad=0.04)

        # Prediction
        ax_pred = fig.add_subplot(gs[field_idx, t_idx * 3 + 1])
        im_pred = ax_pred.imshow(pred, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
        ax_pred.set_title(f'Prediction (t={t+T_in})', fontsize=10)
        ax_pred.axis('off')
        plt.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)

        # Absolute Error
        ax_err = fig.add_subplot(gs[field_idx, t_idx * 3 + 2])
        im_err = ax_err.imshow(error, cmap='hot', origin='lower')
        ax_err.set_title(f'Abs Error (t={t+T_in})', fontsize=10)
        ax_err.axis('off')
        plt.colorbar(im_err, ax=ax_err, fraction=0.046, pad=0.04)

plt.suptitle('Walrus Navier-Stokes Predictions: Comparison with Ground Truth', fontsize=16, y=0.995)
plt.savefig('/Users/Vicky/Documents/UKAEA/Code/Foundation_Models/walrus/demo_notebooks/navier_stokes_comparison.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved comparison plot to: demo_notebooks/navier_stokes_comparison.png")

# %%
# =============================================================================
# STEP 12: Temporal Evolution Plots
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Temporal Evolution of Prediction Errors', fontsize=16)

for field_idx, field_name in enumerate(real_field_names):
    ax = axes[field_idx]

    # Compute error metrics over time
    mse_over_time = []
    mae_over_time = []

    for t in range(T_out):
        pred = y_pred_real[0, t, :, :, 0, field_idx].cpu().numpy()
        ref = y_ref_real[0, t, :, :, 0, field_idx].cpu().numpy()

        mse_t = np.mean((pred - ref) ** 2)
        mae_t = np.mean(np.abs(pred - ref))

        mse_over_time.append(mse_t)
        mae_over_time.append(mae_t)

    # Plot
    timesteps = np.arange(T_in, T_in + T_out)
    ax2 = ax.twinx()

    line1 = ax.plot(timesteps, mse_over_time, 'b-o', label='MSE', linewidth=2, markersize=6)
    line2 = ax2.plot(timesteps, mae_over_time, 'r-s', label='MAE', linewidth=2, markersize=6)

    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('MSE', color='b', fontsize=12)
    ax.set_title(f'{field_name}', fontsize=14, fontweight='bold')
    ax.tick_params(axis='y', labelcolor='b')
    ax.grid(True, alpha=0.3)

    ax2.set_ylabel('MAE', color='r', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='r')

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left')

plt.tight_layout()
plt.savefig('/Users/Vicky/Documents/UKAEA/Code/Foundation_Models/walrus/demo_notebooks/navier_stokes_temporal_errors.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved temporal error plot to: demo_notebooks/navier_stokes_temporal_errors.png")

# %%
# =============================================================================
# STEP 13: Statistical Distribution Comparison
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Statistical Distribution: Predictions vs Ground Truth', fontsize=16)

for field_idx, field_name in enumerate(real_field_names):
    ax = axes[field_idx]

    # Flatten all timesteps
    pred_all = y_pred_real[0, :, :, :, 0, field_idx].cpu().numpy().flatten()
    ref_all = y_ref_real[0, :, :, :, 0, field_idx].cpu().numpy().flatten()

    # Create 2D histogram
    hist, xedges, yedges = np.histogram2d(ref_all, pred_all, bins=100)

    # Plot
    im = ax.imshow(hist.T, origin='lower', aspect='auto', cmap='viridis',
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

    # Add diagonal line (perfect prediction)
    min_val = min(xedges[0], yedges[0])
    max_val = max(xedges[-1], yedges[-1])
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    ax.set_xlabel('Ground Truth', fontsize=12)
    ax.set_ylabel('Prediction', fontsize=12)
    ax.set_title(f'{field_name} (R²={metrics_per_field[field_name]["R2"]:.4f})', fontsize=14)
    ax.legend(loc='upper left')
    plt.colorbar(im, ax=ax, label='Count')

plt.tight_layout()
plt.savefig('/Users/Vicky/Documents/UKAEA/Code/Foundation_Models/walrus/demo_notebooks//navier_stokes_scatter.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved scatter plot to: demo_notebooks/navier_stokes_scatter.png")

print(f"\n" + "="*60)
print("Analysis complete! Check the generated plots:")
print("  1. navier_stokes_comparison.png - Spatial comparisons")
print("  2. navier_stokes_temporal_errors.png - Error evolution")
print("  3. navier_stokes_scatter.png - Statistical distributions")
print("="*60)

# %%
