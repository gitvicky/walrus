"""
ShearFlow_NS_ZSL.py

Load Shear Flow data from The Well format and run Walrus inference
without modifying the codebase.
"""
# %% 
import torch
import h5py
import copy
import numpy as np
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
# STEP 1: Load Shear Flow HDF5 file from The Well format
# =============================================================================

print("="*60)
print("Loading Shear Flow data from The Well format")
print("="*60)

# Configuration
reynolds = '1e5'
schmidt = '2e0'
data_loc = '/Users/Vicky/Documents/UKAEA/Data/The_Well/datasets/shear_flow/data/valid/'
file = f'shear_flow_Reynolds_{reynolds}_Schmidt_{schmidt}.hdf5'
hdf5_path = data_loc + file

print(f"\nLoading file: {file}")

with h5py.File(hdf5_path, 'r') as f:
    # Load dimensions
    Nx, Ny = f['dimensions/x'].shape[0], f['dimensions/y'].shape[0]
    x = np.array(f['dimensions/x'])
    y = np.array(f['dimensions/y'])
    time = np.array(f['dimensions/time'])
    dt = time[1] - time[0]
    
    # Load one trajectory for this example (you can loop over all trajectories)
    traj_idx = 0
    
    # Load fields: velocity [Nt, Nx, Ny, 2], pressure [Nt, Nx, Ny], tracer [Nt, Nx, Ny]
    velocity = torch.tensor(f['t1_fields/velocity'][traj_idx], dtype=torch.float32)  # [Nt, Nx, Ny, 2]
    pressure = torch.tensor(f['t0_fields/pressure'][traj_idx], dtype=torch.float32)  # [Nt, Nx, Ny]
    tracer = torch.tensor(f['t0_fields/tracer'][traj_idx], dtype=torch.float32)      # [Nt, Nx, Ny]
    
    # Load scalars
    reynolds_scalar = f['scalars/Reynolds'][()]
    schmidt_scalar = f['scalars/Schmidt'][()]
    
    # Boundary condition type
    bc_type_map = {"WALL": 0, "OPEN": 1, "PERIODIC": 2}
    
    # Check what boundary conditions exist
    bc_keys = list(f['boundary_conditions'].keys())
    print(f"\nAvailable boundary conditions: {bc_keys}")
    
    # Get boundary condition type from first available BC
    if bc_keys:
        bc_group_name = bc_keys[0]
        bc_type = f[f'boundary_conditions/{bc_group_name}'].attrs.get('bc_type', 'PERIODIC')
    else:
        bc_type = 'PERIODIC'  # Default for shear flow
    
    bc_code = bc_type_map.get(bc_type, 2)  # Default to PERIODIC
    
    print(f"\nLoaded Shear Flow data:")
    print(f"  Velocity shape: {velocity.shape}  # [Nt, Nx, Ny, 2]")
    print(f"  Pressure shape: {pressure.shape}  # [Nt, Nx, Ny]")
    print(f"  Tracer shape: {tracer.shape}      # [Nt, Nx, Ny]")
    print(f"  Boundary conditions: {bc_type}")
    print(f"  Reynolds number: {reynolds_scalar}")
    print(f"  Schmidt number: {schmidt_scalar}")
    print(f"  Spatial grid: {Nx} × {Ny}")
    print(f"  Time steps: {velocity.shape[0]}")
    print(f"  dt: {dt}")

print(f"\n✓ Loaded Shear Flow data successfully")

# %% 
# =============================================================================
# STEP 2: Prepare data in Walrus format
# =============================================================================

print("\n" + "="*60)
print("Preparing data in Walrus format")
print("="*60)

# Split into input and output timesteps
T_in = 6    # Number of input timesteps
T_out = 10  # Number of output timesteps to predict

# Extract velocity components
u = velocity[..., 0]  # [Nt, Nx, Ny]
v = velocity[..., 1]  # [Nt, Nx, Ny]
velocity_z = torch.zeros_like(u)

# For 2D simulations, stack fields: [velocity_x, velocity_y, tracer, pressure]
# Stack all fields: [Nt, Nx, Ny, 4]
all_fields = torch.stack([u, v, velocity_z, tracer, pressure], dim=-1)

# Add depth dimension (D=1) for tensor format consistency
all_fields = all_fields.unsqueeze(-2)  # [Nt, Nx, Ny, 1, 5]

# Split into input and output
Nt_total = all_fields.shape[0]
if T_in + T_out > Nt_total:
    print(f"Warning: T_in ({T_in}) + T_out ({T_out}) > Nt_total ({Nt_total})")
    print(f"Adjusting T_out to {Nt_total - T_in}")
    T_out = Nt_total - T_in

input_fields = all_fields[:T_in].unsqueeze(0)   # [1, T_in, Nx, Ny, 1, 5]
output_fields = all_fields[T_in:T_in+T_out].unsqueeze(0)  # [1, T_out, Nx, Ny, 1, 5]

print(f"\nPrepared fields:")
print(f"  Input shape: {input_fields.shape}  # [B=1, T_in={T_in}, H={Nx}, W={Ny}, D=1, C=5]")
print(f"  Output shape: {output_fields.shape}  # [B=1, T_out={T_out}, H={Nx}, W={Ny}, D=1, C=5]")

# %% 
# =============================================================================
# STEP 3: Create field index mapping
# =============================================================================

print("\n" + "="*60)
print("Creating field index mapping")
print("="*60)

# Map fields to pretrained Walrus embeddings:
# - velocity_x → index 4 (pretrained)
# - velocity_y → index 5 (pretrained)
# - tracer (passive scalar) → index 7 (pretrained)
# - pressure → index 3 (pretrained)

field_indices = torch.tensor([4, 5, 6, 7, 3])  # [velocity_x, velocity_y, tracer, pressure]

# Padded field mask: all fields are real (no padding)
padded_field_mask = torch.tensor([True, True, False, True, True])

print(f"\nField mapping:")
field_names_list = ['velocity_x', 'velocity_y', 'velocity_z', 'tracer', 'pressure']
for i, (idx, name, is_real) in enumerate(zip(field_indices, field_names_list, padded_field_mask)):
    status = "real" if is_real else "padding"
    print(f"  Channel {i}: {name:30s} → embedding {idx:2d} ({status})")

# %% 
# =============================================================================
# STEP 4: Create metadata
# =============================================================================

print("\n" + "="*60)
print("Creating metadata")
print("="*60)

metadata = WellMetadata(
    dataset_name="shear_flow",
    n_spatial_dims=2,  # 2D simulation
    
    # Field organization by rank:
    # Rank 0 (scalars): pressure, tracer
    # Rank 1 (vectors): velocity_x, velocity_y
    field_names={
        0: ['pressure', 'tracer'],
        1: ['velocity_x', 'velocity_y', 'velocity_z'],
        2: []
    },
    
    spatial_resolution=(Nx, Ny),  # 2D grid size
    scalar_names=[],
    constant_field_names={0: [], 1: [], 2: []},
    constant_scalar_names=[],
    boundary_condition_types=[],
    n_files=[],
    n_trajectories_per_file=[],
    n_steps_per_trajectory=[],
    grid_type='cartesian'
)

print(f"✓ Metadata created for {metadata.dataset_name}")

# %% 
# =============================================================================
# STEP 5: Create the batch dictionary (Walrus format)
# =============================================================================

print("\n" + "="*60)
print("Creating batch dictionary")
print("="*60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

shear_flow_batch = {
    # Input fields: [B, T_in, H, W, D, C]
    "input_fields": input_fields.to(device),
    
    # Output fields: [B, T_out, H, W, D, C]
    "output_fields": output_fields.to(device),
    
    # Constant fields: [B, H, W, C_const] - none in this case
    "constant_fields": torch.empty(1, Nx, Ny, 1, 0, device=device),

    
    # Boundary conditions: [B, 3, 2]
    # [x_lower, x_upper], [y_lower, y_upper], [z_lower, z_upper]
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

print(f"✓ Created Walrus batch dictionary")
print(f"  Device: {device}")
print(f"  Ready for model inference!")

# %% 
# =============================================================================
# STEP 6: Load Walrus model
# =============================================================================

print("\n" + "="*60)
print("Loading Walrus model")
print("="*60)

# Set paths to your downloaded model files
checkpoint_base_path = "./checkpoints/"
config_base_path = "./configs/"

checkpoint_path = f"{checkpoint_base_path}/walrus.pt"
checkpoint_config_path = f"{config_base_path}/extended_config.yaml"

# Load checkpoint and config
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["app"]["model"]
config = OmegaConf.load(checkpoint_config_path)

# Get field mapping
field_to_index_map = config.data.field_index_map_override
new_field_to_index_map = dict(field_to_index_map)

# Verify our field indices are in the pretrained model
required_indices = [4, 5, 6, 7, 3]  # velocity_x, velocity_y, tracer, pressure
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

print("\n" + "="*60)
print("Setting up helper objects")
print("="*60)

formatter = ChannelsFirstWithTimeFormatter()
revin = instantiate(config.trainer.revin)()

print(f"✓ Helper objects initialized")

# =============================================================================
# STEP 8: Define rollout function
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
# STEP 9: Run inference on Shear Flow data
# =============================================================================

with torch.no_grad():
    print(f"\n" + "="*60)
    print("Running Walrus inference on Shear Flow data")
    print("="*60)
    
    y_pred, y_ref = rollout_model(
        model,
        revin,
        shear_flow_batch,
        formatter,
        max_rollout_steps=T_out,
        device=device,
    )
    
    print(f"\n✓ Prediction complete!")
    print(f"  Prediction shape: {y_pred.shape}")
    print(f"  Reference shape: {y_ref.shape}")
    
    # All fields are real (no padding in this case)
    y_pred_real = y_pred
    y_ref_real = y_ref
    
    real_field_names = ['velocity_x', 'velocity_y', 'tracer', 'pressure']
    
    print(f"\n  Fields: {real_field_names}")
    print(f"\n" + "="*60)
    print("You can now analyze the predictions!")
    print("="*60)
    print(f"\nExamples:")
    print(f"  - Extract velocity_x: velocity_x_pred = y_pred_real[0, :, :, :, 0, 0]")
    print(f"  - Extract tracer:     tracer_pred = y_pred_real[0, :, :, :, 0, 2]")
    print(f"  - Extract pressure:   pressure_pred = y_pred_real[0, :, :, :, 0, 3]")
    print(f"  - Compute MSE:        mse = (y_pred_real - y_ref_real).pow(2).mean()")
    
    # Compute some basic metrics
    mse = (y_pred_real - y_ref_real).pow(2).mean()
    mse_per_field = (y_pred_real - y_ref_real).pow(2).mean(dim=(0, 1, 2, 3, 4))
    
    print(f"\n" + "="*60)
    print("Basic Metrics")
    print("="*60)
    print(f"  Overall MSE: {mse.item():.6e}")
    print(f"\n  MSE per field:")
    for i, name in enumerate(real_field_names):
        print(f"    {name:15s}: {mse_per_field[i].item():.6e}")

# %%
# =============================================================================
# STEP 10: Optional - Visualize results
# =============================================================================

try:
    import matplotlib.pyplot as plt
    
    print(f"\n" + "="*60)
    print("Generating visualization")
    print("="*60)
    
    # Select a timestep to visualize
    t_idx = T_out // 2  # Middle timestep
    
    # Extract fields for visualization
    vel_x_pred = y_pred_real[0, t_idx, :, :, 0, 0].cpu().numpy()
    vel_y_pred = y_pred_real[0, t_idx, :, :, 0, 1].cpu().numpy()
    tracer_pred = y_pred_real[0, t_idx, :, :, 0, 2].cpu().numpy()
    pressure_pred = y_pred_real[0, t_idx, :, :, 0, 3].cpu().numpy()
    
    vel_x_ref = y_ref_real[0, t_idx, :, :, 0, 0].cpu().numpy()
    vel_y_ref = y_ref_real[0, t_idx, :, :, 0, 1].cpu().numpy()
    tracer_ref = y_ref_real[0, t_idx, :, :, 0, 2].cpu().numpy()
    pressure_ref = y_ref_real[0, t_idx, :, :, 0, 3].cpu().numpy()
    
    # Compute velocity magnitude
    vel_mag_pred = np.sqrt(vel_x_pred**2 + vel_y_pred**2)
    vel_mag_ref = np.sqrt(vel_x_ref**2 + vel_y_ref**2)
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    # Row 1: Velocity magnitude
    im0 = axes[0, 0].imshow(vel_mag_ref, origin='lower', cmap='viridis', aspect='auto')
    axes[0, 0].set_title(f'Velocity Magnitude (Reference) at t={t_idx}')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].imshow(vel_mag_pred, origin='lower', cmap='viridis', aspect='auto')
    axes[0, 1].set_title(f'Velocity Magnitude (Predicted) at t={t_idx}')
    plt.colorbar(im1, ax=axes[0, 1])
    
    im2 = axes[0, 2].imshow(np.abs(vel_mag_pred - vel_mag_ref), origin='lower', cmap='Reds', aspect='auto')
    axes[0, 2].set_title('Absolute Error')
    plt.colorbar(im2, ax=axes[0, 2])
    
    # Row 2: Tracer field
    im3 = axes[1, 0].imshow(tracer_ref, origin='lower', cmap='plasma', aspect='auto')
    axes[1, 0].set_title('Tracer (Reference)')
    axes[1, 0].set_ylabel('y')
    plt.colorbar(im3, ax=axes[1, 0])
    
    im4 = axes[1, 1].imshow(tracer_pred, origin='lower', cmap='plasma', aspect='auto')
    axes[1, 1].set_title('Tracer (Predicted)')
    plt.colorbar(im4, ax=axes[1, 1])
    
    im5 = axes[1, 2].imshow(np.abs(tracer_pred - tracer_ref), origin='lower', cmap='Reds', aspect='auto')
    axes[1, 2].set_title('Absolute Error')
    plt.colorbar(im5, ax=axes[1, 2])
    
    # Row 3: Pressure field
    im6 = axes[2, 0].imshow(pressure_ref, origin='lower', cmap='RdBu_r', aspect='auto')
    axes[2, 0].set_title('Pressure (Reference)')
    axes[2, 0].set_xlabel('x')
    axes[2, 0].set_ylabel('y')
    plt.colorbar(im6, ax=axes[2, 0])
    
    im7 = axes[2, 1].imshow(pressure_pred, origin='lower', cmap='RdBu_r', aspect='auto')
    axes[2, 1].set_title('Pressure (Predicted)')
    axes[2, 1].set_xlabel('x')
    plt.colorbar(im7, ax=axes[2, 1])
    
    im8 = axes[2, 2].imshow(np.abs(pressure_pred - pressure_ref), origin='lower', cmap='Reds', aspect='auto')
    axes[2, 2].set_title('Absolute Error')
    axes[2, 2].set_xlabel('x')
    plt.colorbar(im8, ax=axes[2, 2])
    
    plt.tight_layout()
    plt.savefig('shear_flow_walrus_predictions.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: shear_flow_walrus_predictions.png")
    plt.show()
    
except ImportError:
    print("\nMatplotlib not available - skipping visualization")
except Exception as e:
    print(f"\nVisualization error: {e}")

print(f"\n" + "="*60)
print("Shear Flow inference complete!")
print("="*60)

# %%
