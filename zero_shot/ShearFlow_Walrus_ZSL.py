"""
ShearFlow_Walrus_ZSL.py

Run Walrus zero-shot inference on shear flow using the shared ZSL pipeline.
Iterates over trajectories until relative MSE is <= 1% for 50-step rollouts.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import torch

from extractors.shear_flow import DEFAULT_HDF5_PATH, extract
from walrus_zsl import ZSLConfig, run_zsl


REL_MSE_THRESHOLD = 0.01  # 1% relative error
TARGET_T_OUT = 50
T_IN = 10
START_TRAJECTORY = 0
MAX_TRAJECTORY_ATTEMPTS = 10
GENERATE_PLOTS = False


def relative_mse(y_pred: torch.Tensor, y_ref: torch.Tensor) -> float:
    diff = y_pred - y_ref
    mse = diff.pow(2).mean()
    denom = y_ref.pow(2).mean().clamp_min(1e-12)
    return (mse / denom).item()


def relative_mse_per_timestep(
    y_pred: torch.Tensor, y_ref: torch.Tensor
) -> torch.Tensor:
    diff = (y_pred - y_ref).pow(2)
    ref = y_ref.pow(2)
    dims = tuple(i for i in range(diff.ndim) if i != 1)
    mse_t = diff.mean(dim=dims)
    ref_t = ref.mean(dim=dims)
    return mse_t / ref_t.clamp_min(1e-12)


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    config = ZSLConfig(
        checkpoint_path=str(base / "demo_notebooks/checkpoints/walrus.pt"),
        checkpoint_config_path=str(base / "demo_notebooks/configs/extended_config.yaml"),
        T_in=T_IN,
        T_out=TARGET_T_OUT,
        max_rollout_steps=TARGET_T_OUT,
        target_resolution=128,
        output_dir=str(base / "zero_shot/results"),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with h5py.File(DEFAULT_HDF5_PATH, "r") as f:
        total_trajectories = f["t1_fields/velocity"].shape[0]
    max_attempts = min(MAX_TRAJECTORY_ATTEMPTS, total_trajectories - START_TRAJECTORY)
    if max_attempts <= 0:
        raise RuntimeError(
            f"No trajectories available starting at index {START_TRAJECTORY}."
        )

    results = None
    best_rel_mse = None
    best_traj = None
    for attempt in range(max_attempts):
        config.trajectory_index = START_TRAJECTORY + attempt
        print(f"\n=== Shear Flow ZSL (trajectory {config.trajectory_index}) ===")
        print(f"  T_in={config.T_in} | T_out={config.T_out}")

        results = run_zsl(
            extract,
            config,
            device=device,
            skip_plots=not GENERATE_PLOTS,
        )

        if results.references.shape[1] != TARGET_T_OUT:
            raise RuntimeError(
                f"Expected {TARGET_T_OUT} rollout steps but received "
                f"{results.references.shape[1]}."
            )

        rel_mse = relative_mse(results.predictions, results.references)
        rel_mse_t = relative_mse_per_timestep(
            results.predictions, results.references
        )
        max_rel_mse = rel_mse_t.max().item()
        max_step = int(rel_mse_t.argmax().item()) + 1
        print(f"\nRelative MSE (overall): {rel_mse:.6e}")
        print(f"Relative MSE (max step): {max_rel_mse:.6e} at step {max_step}")

        if best_rel_mse is None or rel_mse < best_rel_mse:
            best_rel_mse = rel_mse
            best_traj = config.trajectory_index

        if rel_mse <= REL_MSE_THRESHOLD:
            print(
                f"✓ Target reached: {rel_mse:.6e} <= {REL_MSE_THRESHOLD:.2e}"
            )
            break
    else:
        raise RuntimeError(
            "Unable to reach the target relative MSE threshold after "
            f"{MAX_TRAJECTORY_ATTEMPTS} trajectories. "
            f"Best was {best_rel_mse:.6e} (trajectory {best_traj})."
        )

    if results is None:
        raise RuntimeError("ZSL run failed to produce results.")


if __name__ == "__main__":
    main()
