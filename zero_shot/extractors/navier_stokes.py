"""
Navier-Stokes data extractor for Walrus ZSL.

This extracts velocity (u, v) and pressure from an HDF5 file produced by
the Well conversion pipeline.  Because the simulation is 2D, velocity_z
is included as a zero-padding channel (required by the 3D-native model).

Usage
-----
    from walrus_zsl import run_zsl, ZSLConfig
    from extractors.navier_stokes import extract

    config = ZSLConfig(...)
    results = run_zsl(extract, config)
"""

import h5py
import torch

from walrus_zsl import (
    BC_PERIODIC,
    ExtractedData,
    FieldSpec,
    ZSLConfig,
)


# Default path — override via the ``hdf5_path`` key in the extractor or
# change this constant.
DEFAULT_HDF5_PATH = (
    "/Users/Vicky/Documents/UKAEA/Code/Foundation_Models/walrus/"
    "demo_notebooks/converted_data/navier_stokes_spectral_id_n5.hdf5"
)


def extract(
    config: ZSLConfig,
    hdf5_path: str = DEFAULT_HDF5_PATH,
    trajectory_index: int | None = None,
) -> ExtractedData:
    """Load Navier-Stokes data and return an ``ExtractedData`` bundle.

    Parameters
    ----------
    config : ZSLConfig
        Framework configuration (``trajectory_index`` is used if
        *trajectory_index* is not given explicitly).
    hdf5_path : str
        Path to the HDF5 file.
    trajectory_index : int or None
        Which trajectory to load.  Falls back to ``config.trajectory_index``.
    """
    traj = trajectory_index if trajectory_index is not None else config.trajectory_index

    with h5py.File(hdf5_path, "r") as f:
        velocity = torch.tensor(f["t1_fields/velocity"][traj], dtype=torch.float32)
        pressure = torch.tensor(f["t0_fields/pressure"][traj], dtype=torch.float32)

    # velocity: [Nt, Nx, Ny, 2]  →  split into u, v
    u = velocity[..., 0]       # [Nt, Nx, Ny]
    v = velocity[..., 1]       # [Nt, Nx, Ny]
    w = torch.zeros_like(u)    # padding for velocity_z
    # pressure: [Nt, Nx, Ny]

    # Stack into [Nt, Nx, Ny, C=4]
    fields = torch.stack([u, v, w, pressure], dim=-1)

    field_specs = [
        FieldSpec(name="velocity_x",  walrus_index=4,  tensor_rank=1),
        FieldSpec(name="velocity_y",  walrus_index=5,  tensor_rank=1),
        FieldSpec(name="velocity_z",  walrus_index=6,  tensor_rank=1, is_padding=True),
        FieldSpec(name="pressure",    walrus_index=3,  tensor_rank=0),
    ]

    # Periodic BCs in all three directions (z is dummy for the 3D wrapper)
    boundary_conditions = [
        (BC_PERIODIC, BC_PERIODIC),  # x
        (BC_PERIODIC, BC_PERIODIC),  # y
        (BC_PERIODIC, BC_PERIODIC),  # z (padding dim)
    ]

    return ExtractedData(
        fields=fields,
        field_specs=field_specs,
        n_spatial_dims=3,              # Report 3D because we pad D=1
        boundary_conditions=boundary_conditions,
        dataset_name="navier_stokes_spectral",
        target_resolution=128,         # Resize to 128x128
    )
