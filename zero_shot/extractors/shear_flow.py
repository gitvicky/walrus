"""
Shear Flow data extractor for Walrus ZSL.

This extracts velocity (u, v), tracer, and pressure from a Well-format
HDF5 file. The simulation is natively 2D, so velocity_z is added as a
zero-padding channel to match the pretrained 3D-native model.

Usage
-----
    from walrus_zsl import run_zsl, ZSLConfig
    from extractors.shear_flow import extract

    config = ZSLConfig(T_in=5, T_out=20, ...)
    results = run_zsl(extract, config)
"""

import h5py
import torch
import torch.nn.functional as F
import yaml

from walrus_zsl import (
    BC_PERIODIC,
    ExtractedData,
    FieldSpec,
    VALID_SPATIAL_SIZES,
    ZSLConfig,
)

DEFAULT_HDF5_PATH = (
    "/Users/Vicky/Documents/UKAEA/Data/The_Well/datasets/"
    "shear_flow/data/valid/shear_flow_Reynolds_1e5_Schmidt_2e0.hdf5"
)

STATS_PATH = (
    "/Users/Vicky/Documents/UKAEA/Data/The_Well/datasets/shear_flow/stats.yaml"
)
USE_STATS_NORMALIZATION = False  # Training uses samplewise normalization on raw data.
NORMALIZATION_MODE = "zscore"  # "zscore" or "rms"
_STATS_CACHE: dict | None = None


def _load_stats() -> dict:
    global _STATS_CACHE
    if _STATS_CACHE is None:
        with open(STATS_PATH, "r", encoding="utf-8") as f:
            _STATS_CACHE = yaml.safe_load(f)
    return _STATS_CACHE


def _apply_normalization(
    u: torch.Tensor,
    v: torch.Tensor,
    tracer: torch.Tensor,
    pressure: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    stats = _load_stats()
    if NORMALIZATION_MODE == "zscore":
        u = (u - float(stats["mean"]["velocity"][0])) / float(stats["std"]["velocity"][0])
        v = (v - float(stats["mean"]["velocity"][1])) / float(stats["std"]["velocity"][1])
        tracer = (tracer - float(stats["mean"]["tracer"])) / float(stats["std"]["tracer"])
        pressure = (pressure - float(stats["mean"]["pressure"])) / float(stats["std"]["pressure"])
    elif NORMALIZATION_MODE == "rms":
        u = u / float(stats["rms"]["velocity"][0])
        v = v / float(stats["rms"]["velocity"][1])
        tracer = tracer / float(stats["rms"]["tracer"])
        pressure = pressure / float(stats["rms"]["pressure"])
    else:
        raise ValueError(f"Unknown NORMALIZATION_MODE: {NORMALIZATION_MODE}")
    return u, v, tracer, pressure


def extract(
    config: ZSLConfig,
    hdf5_path: str = DEFAULT_HDF5_PATH,
    trajectory_index: int | None = None,
) -> ExtractedData:
    """Load Shear Flow data and return an ``ExtractedData`` bundle."""
    traj = trajectory_index if trajectory_index is not None else config.trajectory_index

    with h5py.File(hdf5_path, "r") as f:
        velocity = torch.tensor(f["t1_fields/velocity"][traj], dtype=torch.float32)
        pressure = torch.tensor(f["t0_fields/pressure"][traj], dtype=torch.float32)
        tracer   = torch.tensor(f["t0_fields/tracer"][traj],   dtype=torch.float32)

    u = velocity[..., 0]  # [Nt, Nx, Ny]
    v = velocity[..., 1]  # [Nt, Nx, Ny]
    w = torch.zeros_like(u)  # padding for velocity_z

    if USE_STATS_NORMALIZATION:
        u, v, tracer, pressure = _apply_normalization(u, v, tracer, pressure)

    target_resolution = config.target_resolution
    if target_resolution is not None:
        def _resize(field: torch.Tensor) -> torch.Tensor:
            # [Nt, H, W] -> [Nt, 1, H, W] for interpolate
            field = field.unsqueeze(1)
            field = F.interpolate(
                field,
                size=(target_resolution, target_resolution),
                mode="bilinear",
                align_corners=False,
            )
            return field.squeeze(1)

        u = _resize(u)
        v = _resize(v)
        w = torch.zeros_like(u)
        tracer = _resize(tracer)
        pressure = _resize(pressure)

    # Stack with scalar fields first to match Well/InflatedWellDataset ordering.
    # Order: tracer, pressure, velocity_x, velocity_y, velocity_z
    fields = torch.stack([tracer, pressure, u, v, w], dim=-1)
    # Pad to 3D (D=1) to match training-time cartesian padding.
    fields = fields.unsqueeze(-2)  # [Nt, Nx, Ny, 1, C]

    field_specs = [
        FieldSpec(name="tracer",     walrus_index=54, tensor_rank=0),
        FieldSpec(name="pressure",   walrus_index=3,  tensor_rank=0),
        FieldSpec(name="velocity_x", walrus_index=4,  tensor_rank=1),
        FieldSpec(name="velocity_y", walrus_index=5,  tensor_rank=1),
        FieldSpec(name="velocity_z", walrus_index=6,  tensor_rank=1, is_padding=True),
    ]

    boundary_conditions = [
        (BC_PERIODIC, BC_PERIODIC),  # x
        (BC_PERIODIC, BC_PERIODIC),  # y
        (BC_PERIODIC, BC_PERIODIC),  # z (padding dim)
    ]

    Nx, Ny = u.shape[1], u.shape[2]
    if target_resolution is None and (
        Nx not in VALID_SPATIAL_SIZES or Ny not in VALID_SPATIAL_SIZES
    ):
        target = max(Nx, Ny)
        target_resolution = min(VALID_SPATIAL_SIZES, key=lambda s: abs(s - target))

    return ExtractedData(
        fields=fields,
        field_specs=field_specs,
        n_spatial_dims=3,
        boundary_conditions=boundary_conditions,
        dataset_name="shear_flow",
        target_resolution=target_resolution,
    )
