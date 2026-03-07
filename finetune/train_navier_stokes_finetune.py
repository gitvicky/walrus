"""
Finetuning script for Walrus model on Navier-Stokes Spectral dataset.

This script demonstrates how to finetune a pretrained Walrus model on novel physics data.
The script can be launched on a single GPU or across multiple GPUs using torchrun.

Usage:
    Single GPU:
        python train_navier_stokes_finetune.py

    Multi-GPU (DDP):
        torchrun --nproc_per_node=4 train_navier_stokes_finetune.py
"""

import copy
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import h5py
import hydra
import torch
import torch.nn.functional as F
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from the_well.data.datasets import WellMetadata
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary

from walrus.data.well_to_multi_transformer import ChannelsFirstWithTimeFormatter
from walrus.optim.optim_utils import build_param_groups
from walrus.trainer.checkpoints import CheckPointLoader
from walrus.trainer.training import Trainer
from walrus.utils.distribution_utils import configure_distribution, distribute_model
from walrus.utils.experiment_utils import (
    align_checkpoint_with_field_to_index_map,
    configure_experiment,
)

logger = logging.getLogger("walrus_finetune")
logging.basicConfig(level=logging.INFO)


class NavierStokesSpectralDataset(Dataset):
    """Custom PyTorch dataset for Navier-Stokes Spectral data in HDF5 format."""

    def __init__(
        self,
        hdf5_path: str,
        split: str = "train",
        n_steps_input: int = 4,
        n_steps_output: int = 20,
        target_size: int = 128,
        train_split: float = 0.7,
        val_split: float = 0.15,
        # test_split: float = 0.15
    ):
        """
        Initialize Navier-Stokes dataset.

        Args:
            hdf5_path: Path to the HDF5 file containing Navier-Stokes data
            split: One of 'train', 'valid', or 'test'
            n_steps_input: Number of input timesteps
            n_steps_output: Number of output timesteps to predict
            target_size: Spatial resolution to resize to (must be compatible with model)
            train_split: Fraction of trajectories for training
            val_split: Fraction of trajectories for validation
        """
        self.hdf5_path = hdf5_path
        self.split = split
        self.n_steps_input = n_steps_input
        self.n_steps_output = n_steps_output
        self.target_size = target_size

        # Load metadata from HDF5
        with h5py.File(hdf5_path, 'r') as f:
            # Get number of trajectories
            velocity_data = f['t1_fields/velocity']
            n_trajectories = velocity_data.shape[0]
            self.n_timesteps = velocity_data.shape[1]

            # Get boundary condition info
            bc_type_map = {"WALL": 0, "OPEN": 1, "PERIODIC": 2}
            bc_x = f['boundary_conditions/x_periodic'].attrs['bc_type']
            bc_y = f['boundary_conditions/y_periodic'].attrs['bc_type']
            self.bc_code = bc_type_map[bc_x]

        # Split trajectories into train/val/test
        train_end = int(n_trajectories * train_split)
        val_end = int(n_trajectories * (train_split + val_split))

        if split == "train":
            self.trajectory_indices = list(range(0, train_end))
        elif split == "valid":
            self.trajectory_indices = list(range(train_end, val_end))
        else:  # test
            self.trajectory_indices = list(range(val_end, n_trajectories))

        # Create metadata for Walrus format
        self.metadata = WellMetadata(
            dataset_name="navier_stokes_spectral",
            n_spatial_dims=3,  # Walrus expects 3D (we pad D=1)
            field_names={
                0: ['pressure'],
                1: ['velocity_x', 'velocity_y', 'velocity_z'],
                2: []
            },
            spatial_resolution=(target_size, target_size, 1),
            scalar_names=[],
            constant_field_names={0: [], 1: [], 2: []},
            constant_scalar_names=[],
            boundary_condition_types=[],
            n_files=[],
            n_trajectories_per_file=[],
            n_steps_per_trajectory=[],
            grid_type='cartesian'
        )

        # Field mapping to pretrained Walrus embeddings
        # velocity_x → 4, velocity_y → 5, velocity_z → 6 (padding), pressure → 3
        self.field_indices = torch.tensor([4, 5, 6, 3], dtype=torch.long)
        self.padded_field_mask = torch.tensor([True, True, False, True])

        logger.info(f"Initialized {split} dataset with {len(self.trajectory_indices)} trajectories")

    def __len__(self):
        """Return number of valid samples in the dataset."""
        # Each trajectory can provide multiple samples with sliding windows
        max_start_idx = self.n_timesteps - self.n_steps_input - self.n_steps_output
        return len(self.trajectory_indices) * max_start_idx

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        Returns a batch dictionary in Walrus format.
        """
        # Determine which trajectory and which time window
        max_start_idx = self.n_timesteps - self.n_steps_input - self.n_steps_output
        traj_idx = self.trajectory_indices[idx // max_start_idx]
        time_start = idx % max_start_idx

        # Load data from HDF5
        with h5py.File(self.hdf5_path, 'r') as f:
            # Load velocity [Nt, Nx, Ny, 2] and pressure [Nt, Nx, Ny]
            velocity = torch.tensor(
                f['t1_fields/velocity'][traj_idx, time_start:time_start + self.n_steps_input + self.n_steps_output],
                dtype=torch.float32
            )
            pressure = torch.tensor(
                f['t0_fields/pressure'][traj_idx, time_start:time_start + self.n_steps_input + self.n_steps_output],
                dtype=torch.float32
            )

        # Extract velocity components
        u = velocity[..., 0]  # [Nt, Nx, Ny]
        v = velocity[..., 1]  # [Nt, Nx, Ny]

        # Resize to target size using bilinear interpolation
        u = F.interpolate(u.unsqueeze(1), size=(self.target_size, self.target_size),
                         mode='bilinear', align_corners=False).squeeze(1)
        v = F.interpolate(v.unsqueeze(1), size=(self.target_size, self.target_size),
                         mode='bilinear', align_corners=False).squeeze(1)
        pressure = F.interpolate(pressure.unsqueeze(1), size=(self.target_size, self.target_size),
                                mode='bilinear', align_corners=False).squeeze(1)

        # Create zero padding for velocity_z
        velocity_z = torch.zeros_like(u)

        # Stack all fields: [Nt, Nx, Ny, 4]
        all_fields = torch.stack([u, v, velocity_z, pressure], dim=-1)

        # Add depth dimension (D=1): [Nt, Nx, Ny, 1, 4]
        all_fields = all_fields.unsqueeze(-2)

        # Split into input and output
        input_fields = all_fields[:self.n_steps_input]  # [T_in, H, W, D, C]
        output_fields = all_fields[self.n_steps_input:]  # [T_out, H, W, D, C]

        # Create boundary conditions tensor
        boundary_conditions = torch.tensor(
            [[[self.bc_code, self.bc_code],
              [self.bc_code, self.bc_code],
              [self.bc_code, self.bc_code]]],
            dtype=torch.long
        )

        # Return batch dictionary in Walrus format
        return {
            "input_fields": input_fields,  # [T_in, H, W, D, C]
            "output_fields": output_fields,  # [T_out, H, W, D, C]
            "constant_fields": torch.empty(self.target_size, self.target_size, 1, 0),  # No constants
            "boundary_conditions": boundary_conditions,  # [1, 3, 2]
            "padded_field_mask": self.padded_field_mask,  # [C]
            "field_indices": self.field_indices,  # [C]
            "metadata": self.metadata,
        }


def collate_fn(batch):
    """Custom collate function for Navier-Stokes data."""
    # Batch size
    B = len(batch)

    # Stack all tensors, keeping metadata separate
    metadata = batch[0]["metadata"]  # Same for all samples

    return {
        "input_fields": torch.stack([sample["input_fields"] for sample in batch]),  # [B, T_in, H, W, D, C]
        "output_fields": torch.stack([sample["output_fields"] for sample in batch]),  # [B, T_out, H, W, D, C]
        "constant_fields": torch.stack([sample["constant_fields"] for sample in batch]),  # [B, H, W, D, 0]
        "boundary_conditions": torch.stack([sample["boundary_conditions"] for sample in batch]).squeeze(1),  # [B, 3, 2]
        "padded_field_mask": batch[0]["padded_field_mask"],  # [C] - same for all
        "field_indices": batch[0]["field_indices"],  # [C] - same for all
        "metadata": metadata,
    }


class NavierStokesDataModule:
    """DataModule wrapper for Navier-Stokes dataset to match Walrus interface."""

    def __init__(
        self,
        hdf5_path: str,
        batch_size: int = 8,
        n_steps_input: int = 4,
        n_steps_output: int = 20,
        target_size: int = 128,
        num_workers: int = 4,
        world_size: int = 1,
        rank: int = 0,
    ):
        self.hdf5_path = hdf5_path
        self.batch_size = batch_size
        self.n_steps_input = n_steps_input
        self.n_steps_output = n_steps_output
        self.target_size = target_size
        self.num_workers = num_workers
        self.world_size = world_size
        self.rank = rank

        # Create datasets
        self.train_dataset = NavierStokesSpectralDataset(
            hdf5_path=hdf5_path,
            split="train",
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            target_size=target_size,
        )

        self.val_dataset = NavierStokesSpectralDataset(
            hdf5_path=hdf5_path,
            split="valid",
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            target_size=target_size,
        )

        self.test_dataset = NavierStokesSpectralDataset(
            hdf5_path=hdf5_path,
            split="test",
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            target_size=target_size,
        )

    def train_dataloader(self, rank_override=None):
        """Create training dataloader."""
        if self.world_size > 1:
            sampler = torch.utils.data.DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.rank if rank_override is None else rank_override,
                shuffle=True,
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def val_dataloaders(self, replicas=None, rank=None, full=False):
        """Create validation dataloaders (returns list for compatibility)."""
        if self.world_size > 1:
            sampler = torch.utils.data.DistributedSampler(
                self.val_dataset,
                num_replicas=replicas if replicas else self.world_size,
                rank=rank if rank is not None else self.rank,
                shuffle=False,
            )
        else:
            sampler = None

        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        return [loader]

    def rollout_val_dataloaders(self, replicas=None, rank=None, full=False):
        """Create rollout validation dataloaders (returns list for compatibility)."""
        # For rollout, use batch_size=1 to conserve memory
        if self.world_size > 1:
            sampler = torch.utils.data.DistributedSampler(
                self.val_dataset,
                num_replicas=replicas if replicas else self.world_size,
                rank=rank if rank is not None else self.rank,
                shuffle=False,
            )
        else:
            sampler = None

        loader = DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        return [loader]

    def test_dataloaders(self, replicas=None, rank=None, full=True):
        """Create test dataloaders (returns list for compatibility)."""
        if self.world_size > 1:
            sampler = torch.utils.data.DistributedSampler(
                self.test_dataset,
                num_replicas=replicas if replicas else self.world_size,
                rank=rank if rank is not None else self.rank,
                shuffle=False,
            )
        else:
            sampler = None

        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        return [loader]

    def rollout_test_dataloaders(self, replicas=None, rank=None, full=True):
        """Create rollout test dataloaders (returns list for compatibility)."""
        if self.world_size > 1:
            sampler = torch.utils.data.DistributedSampler(
                self.test_dataset,
                num_replicas=replicas if replicas else self.world_size,
                rank=rank if rank is not None else self.rank,
                shuffle=False,
            )
        else:
            sampler = None

        loader = DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        return [loader]


def load_pretrained_model(
    model: torch.nn.Module,
    checkpoint_path: str,
    field_to_index_map: Dict,
    old_field_index_map: Optional[Dict] = None,
):
    """Load pretrained model weights and align field indices."""
    logger.info(f"Loading pretrained checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_checkpoint = checkpoint["app"]["model"]

    # Align field indices if needed
    if old_field_index_map and field_to_index_map != old_field_index_map:
        model_checkpoint = align_checkpoint_with_field_to_index_map(
            checkpoint_state_dict=model_checkpoint,
            model_state_dict=model.state_dict(),
            checkpoint_field_to_index_map=old_field_index_map,
            model_field_to_index_map=field_to_index_map,
        )

    model.load_state_dict(model_checkpoint, strict=True)
    logger.info("Pretrained model loaded successfully")
    return model


def main():
    """Main training function."""

    # ========================================================================
    # Configuration
    # ========================================================================

    # Paths
    HDF5_DATA_PATH = "/Users/Vicky/Documents/UKAEA/Code/Foundation_Models/walrus/demo_notebooks/converted_data/navier_stokes_spectral_id_n5.hdf5"
    CHECKPOINT_PATH = "/Users/Vicky/Documents/UKAEA/Code/Foundation_Models/walrus/demo_notebooks/checkpoints/walrus.pt"
    CONFIG_PATH = "/Users/Vicky/Documents/UKAEA/Code/Foundation_Models/walrus/demo_notebooks/configs/extended_config.yaml"
    EXPERIMENT_ROOT = "./experiments/navier_stokes_finetune"

    # Training hyperparameters
    BATCH_SIZE = 4
    N_STEPS_INPUT = 4
    N_STEPS_OUTPUT = 1  # Single-step prediction during training
    TARGET_SIZE = 128
    MAX_EPOCHS = 50
    LEARNING_RATE = 1e-5
    NUM_WORKERS = 4

    # Validation settings
    VAL_FREQUENCY = 5  # Validate every N epochs
    ROLLOUT_VAL_FREQUENCY = 10  # Rollout validation every N epochs
    MAX_ROLLOUT_STEPS = 20
    SHORT_VALIDATION_LENGTH = 10

    # Distribution settings
    ENABLE_AMP = True  # Automatic mixed precision
    AMP_TYPE = "float16"
    GRAD_ACC_STEPS = 1
    CLIP_GRADIENT = 1.0

    # ========================================================================
    # Setup distributed training
    # ========================================================================

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_distributed = world_size > 1

    if is_distributed:
        torch.distributed.init_process_group(backend="nccl")
        logger.info(f"Initialized distributed training: rank {rank}/{world_size}")

    # Set device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")

    # ========================================================================
    # Load pretrained config and setup field mapping
    # ========================================================================

    config = OmegaConf.load(CONFIG_PATH)
    old_field_to_index_map = dict(config.data.field_index_map_override)

    # Our fields map to existing pretrained embeddings
    field_to_index_map = {
        "velocity_x": 4,
        "velocity_y": 5,
        "velocity_z": 6,  # Padding, but uses pretrained embedding
        "pressure": 3,
    }

    logger.info(f"Field mapping: {field_to_index_map}")

    # ========================================================================
    # Create datamodule
    # ========================================================================

    logger.info("Creating Navier-Stokes datamodule")
    datamodule = NavierStokesDataModule(
        hdf5_path=HDF5_DATA_PATH,
        batch_size=BATCH_SIZE,
        n_steps_input=N_STEPS_INPUT,
        n_steps_output=N_STEPS_OUTPUT,
        target_size=TARGET_SIZE,
        num_workers=NUM_WORKERS,
        world_size=world_size,
        rank=rank,
    )

    # ========================================================================
    # Create model
    # ========================================================================

    logger.info("Instantiating model")
    n_states = max(field_to_index_map.values()) + 1
    model = instantiate(config.model, n_states=n_states)

    # Load pretrained weights
    model = load_pretrained_model(
        model=model,
        checkpoint_path=CHECKPOINT_PATH,
        field_to_index_map=field_to_index_map,
        old_field_index_map=old_field_to_index_map,
    )

    model = model.to(device)

    if rank == 0:
        summary(model, depth=5)

    # ========================================================================
    # Setup optimizer and scheduler
    # ========================================================================

    logger.info("Setting up optimizer")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0,
    )

    # Learning rate scheduler with warmup
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=MAX_EPOCHS,
        eta_min=LEARNING_RATE / 10,
    )

    # ========================================================================
    # Setup checkpointing
    # ========================================================================

    os.makedirs(EXPERIMENT_ROOT, exist_ok=True)
    checkpoint_folder = Path(EXPERIMENT_ROOT) / "checkpoints"
    viz_folder = Path(EXPERIMENT_ROOT) / "visualizations"
    checkpoint_folder.mkdir(exist_ok=True)
    viz_folder.mkdir(exist_ok=True)

    checkpointer = CheckPointLoader(
        checkpoint_folder=str(checkpoint_folder),
        rank=rank,
    )

    # ========================================================================
    # Setup trainer
    # ========================================================================

    logger.info("Instantiating trainer")
    trainer = Trainer(
        experiment_name="navier_stokes_finetune",
        viz_folder=str(viz_folder),
        formatter=ChannelsFirstWithTimeFormatter,
        model=model,
        datamodule=datamodule,
        revin=instantiate(config.trainer.revin),
        optimizer=optimizer,
        loss_fn=instantiate(config.trainer.loss_fn),
        prediction_type=config.trainer.prediction_type,
        max_epoch=MAX_EPOCHS,
        val_frequency=VAL_FREQUENCY,
        rollout_val_frequency=ROLLOUT_VAL_FREQUENCY,
        max_rollout_steps=MAX_ROLLOUT_STEPS,
        short_validation_length=SHORT_VALIDATION_LENGTH,
        checkpointer=checkpointer,
        num_time_intervals=5,
        skip_checkpointing=False,
        validation_suite=instantiate(config.trainer.validation_suite),
        lr_scheduler=lr_scheduler,
        device=device,
        device_mesh=None,  # No FSDP for simplicity
        distribution_type="ddp" if is_distributed else "local",
        rank=rank,
        world_size=world_size,
        enable_amp=ENABLE_AMP,
        amp_type=AMP_TYPE,
        grad_acc_steps=GRAD_ACC_STEPS,
        clip_gradient=CLIP_GRADIENT,
        wandb_logging=True if rank == 0 else False,
        start_epoch=1,
        lr_scheduler_per_step=False,
    )

    # ========================================================================
    # Initialize wandb logging
    # ========================================================================

    if rank == 0:
        wandb.init(
            project="walrus-navier-stokes-finetune",
            name="ns_spectral_finetune",
            config={
                "batch_size": BATCH_SIZE,
                "n_steps_input": N_STEPS_INPUT,
                "n_steps_output": N_STEPS_OUTPUT,
                "max_epochs": MAX_EPOCHS,
                "learning_rate": LEARNING_RATE,
                "target_size": TARGET_SIZE,
                "world_size": world_size,
            },
        )

    # ========================================================================
    # Train
    # ========================================================================

    logger.info("Starting training")
    trainer.train()

    logger.info("Training complete!")

    if rank == 0:
        wandb.finish()

    if is_distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    # Set torch optimization settings
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True

    main()
