"""
Template data extractor for Walrus ZSL.
=======================================

Copy this file, rename it, and fill in the ``extract`` function for your
dataset.  The framework will call ``extract(config)`` and expects an
``ExtractedData`` object back.

Quick reference — available pretrained field indices
----------------------------------------------------
See ``FIELD_INDEX_REFERENCE`` in ``walrus_zsl.py`` for the full table.
The most commonly used ones are listed below for convenience:

    Core fluid
    ----------
    pressure             3       velocity_x           4
    velocity_y           5       velocity_z           6
    density             28       energy              29

    Thermodynamic
    -------------
    temperature         46       entropy             48
    internal_energy     45       speed_of_sound       8

    Transport / passive
    -------------------
    concentration        9       tracer              54
    buoyancy            53

    Magnetic (Cartesian)
    --------------------
    magnetic_field_x    39       magnetic_field_y    40
    magnetic_field_z    41

    Momentum
    --------
    momentum_x          33       momentum_y          34
    momentum_z          35

    Velocity (polar/spherical)
    --------------------------
    velocity_r          30       velocity_theta      31
    velocity_phi        32

    Generic placeholders
    --------------------
    A                   42       B                   43

    Boundary condition codes
    ------------------------
    BC_WALL = 0     BC_OPEN = 1     BC_PERIODIC = 2

Choosing the right index
~~~~~~~~~~~~~~~~~~~~~~~~
Pick the index whose *physical meaning* is closest to your variable.
If nothing matches well, use the generic ``A`` (42) or ``B`` (43) slots,
though zero-shot quality may degrade since those embeddings had less
training signal.

For 2D simulations mapped to the 3D model, add ``velocity_z`` (index 6)
as a zero-padding channel and mark it ``is_padding=True``.

Boundary conditions
~~~~~~~~~~~~~~~~~~~
Provide one ``(lower, upper)`` pair per spatial dimension of your
*metadata* (not the raw data).  If you report ``n_spatial_dims=3`` for
a 2D dataset (because you add a depth-padding dim), include a third
dummy BC pair — typically ``(BC_PERIODIC, BC_PERIODIC)``.
"""

import h5py
import torch

from walrus_zsl import (
    BC_OPEN,
    BC_PERIODIC,
    BC_WALL,
    ExtractedData,
    FieldSpec,
    ZSLConfig,
)


def extract(config: ZSLConfig) -> ExtractedData:
    """Load your dataset and return an ExtractedData object.

    Parameters
    ----------
    config : ZSLConfig
        The run configuration.  Use ``config.trajectory_index`` to select
        which trajectory to load.

    Returns
    -------
    ExtractedData
    """
    # ── 1. Load raw data ─────────────────────────────────────────────
    #
    # Replace this block with your own data loading logic.
    # The goal is to end up with individual 3-D arrays of shape
    # [Nt, Nx, Ny] (2D) or [Nt, Nx, Ny, Nz] (3D) — one per field.

    hdf5_path = "/path/to/your/data.hdf5"
    traj = config.trajectory_index

    with h5py.File(hdf5_path, "r") as f:
        # Example — adjust keys to match your file layout:
        velocity = torch.tensor(f["t1_fields/velocity"][traj], dtype=torch.float32)
        pressure = torch.tensor(f["t0_fields/pressure"][traj], dtype=torch.float32)
        # temperature = torch.tensor(f["t0_fields/temperature"][traj], dtype=torch.float32)

    # ── 2. Split / reshape into individual field arrays ──────────────
    u = velocity[..., 0]    # [Nt, Nx, Ny]
    v = velocity[..., 1]    # [Nt, Nx, Ny]
    p = pressure            # [Nt, Nx, Ny]

    # If your simulation is 2D but you want to use the 3D pathway
    # (n_spatial_dims=3), add a zero-padding channel for velocity_z:
    # w = torch.zeros_like(u)

    # ── 3. Stack into a single tensor ────────────────────────────────
    # Shape: [Nt, Nx, Ny, C]  (2D)
    #    or: [Nt, Nx, Ny, Nz, C]  (3D)
    fields = torch.stack([u, v, p], dim=-1)

    # ── 4. Define field specs ────────────────────────────────────────
    # One FieldSpec per channel, in the SAME order as the C dimension.
    field_specs = [
        FieldSpec(name="velocity_x", walrus_index=4, tensor_rank=1),
        FieldSpec(name="velocity_y", walrus_index=5, tensor_rank=1),
        # FieldSpec(name="velocity_z", walrus_index=6, tensor_rank=1, is_padding=True),
        FieldSpec(name="pressure",   walrus_index=3, tensor_rank=0),
    ]

    # ── 5. Boundary conditions ───────────────────────────────────────
    # One (lower, upper) pair per spatial dimension.
    boundary_conditions = [
        (BC_PERIODIC, BC_PERIODIC),  # x
        (BC_PERIODIC, BC_PERIODIC),  # y
        # (BC_PERIODIC, BC_PERIODIC),  # z  (uncomment for 3D / padded 3D)
    ]

    # ── 6. Return ────────────────────────────────────────────────────
    return ExtractedData(
        fields=fields,
        field_specs=field_specs,
        n_spatial_dims=2,              # 2 or 3
        boundary_conditions=boundary_conditions,
        dataset_name="my_dataset",
        target_resolution=128,         # Set to None to keep native resolution
    )
