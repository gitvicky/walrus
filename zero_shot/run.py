#!/usr/bin/env python
"""
Walrus Zero-Shot Learning — Runner
===================================

Minimal entry point.  Pick an extractor, set your paths, and go.

Examples
--------
    # Navier-Stokes (default paths)
    python run.py

    # Shear Flow
    python run.py --extractor shear_flow

    # Custom extractor in extractors/my_pde.py
    python run.py --extractor my_pde
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

# Ensure the zero_shot/ directory is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from walrus_zsl import ZSLConfig, run_zsl


def main():
    parser = argparse.ArgumentParser(description="Walrus ZSL runner")
    parser.add_argument(
        "--extractor", "-e", default="navier_stokes",
        help="Name of the extractor module inside extractors/ (without .py)",
    )
    parser.add_argument("--checkpoint", default=None, help="Path to walrus.pt")
    parser.add_argument("--config", default=None, help="Path to extended_config.yaml")
    parser.add_argument("--t-in", type=int, default=4, help="Input timesteps")
    parser.add_argument("--t-out", type=int, default=20, help="Output timesteps")
    parser.add_argument("--max-rollout", type=int, default=200, help="Max rollout steps")
    parser.add_argument("--traj", type=int, default=0, help="Trajectory index")
    parser.add_argument("--output-dir", default="zero_shot/results", help="Output directory")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    # Dynamically import the extractor module
    try:
        mod = importlib.import_module(f"extractors.{args.extractor}")
    except ModuleNotFoundError:
        print(f"ERROR: Could not find extractors/{args.extractor}.py")
        print("Available extractors:")
        for p in sorted(Path(__file__).resolve().parent.glob("extractors/*.py")):
            if p.name != "__init__.py":
                print(f"  {p.stem}")
        sys.exit(1)

    extract_fn = mod.extract

    # Build config
    base = Path(__file__).resolve().parent.parent  # walrus project root
    config = ZSLConfig(
        checkpoint_path=args.checkpoint or str(base / "demo_notebooks/checkpoints/walrus.pt"),
        checkpoint_config_path=args.config or str(base / "demo_notebooks/configs/extended_config.yaml"),
        T_in=args.t_in,
        T_out=args.t_out,
        max_rollout_steps=args.max_rollout,
        trajectory_index=args.traj,
        output_dir=args.output_dir,
    )

    results = run_zsl(extract_fn, config, skip_plots=args.no_plots)
    print(f"\nDone. Overall R²: {results.metrics['__overall__']['R2']:.4f}")


if __name__ == "__main__":
    main()
