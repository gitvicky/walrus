SHEAR FLOW ZERO-SHOT SUMMARY

Goal
- Match paper results for shear_flow zero-shot autoregressive rollout (50 steps) with <1% relative MSE.

Key scripts/files touched
- zero_shot/ShearFlow_Walrus_ZSL.py
- zero_shot/extractors/shear_flow.py
- zero_shot/walrus_zsl.py (inspection only)

Approaches tried and results
1) Match training rollout length and input history
- Set T_out=50 and T_in=10 (per shear_flow config).
- Result: rollout runs but errors were extremely large (relative MSE ~1e19–1e27).

2) Pad to 3D and set boundary conditions
- Added D=1 padding and velocity_z=0 channel.
- Added z periodic BC for padded dimension.
- Result: shape alignment fixed; metrics still huge.

3) Field ordering to match WellDataset/InflatedWellDataset
- Ensured order: tracer, pressure, velocity_x, velocity_y, velocity_z.
- Result: order aligned, errors still huge.

4) Resolution changes
- 64x64: invalid for stride modulation (KeyError).
- 32x32: produced NaNs.
- 128x128: valid, but errors still huge.

5) Dataset stats normalization (stats.yaml)
- Added optional z-score and RMS normalization using stats.yaml.
- Result: with stats normalization, errors remained huge (relative MSE ~1e26).

6) Trajectory guard
- Capped attempts by number of trajectories in HDF5 to avoid IndexError.
- Result: stability improvement only, no effect on MSE.

7) Plotting disabled
- Disabled plots to avoid NaN scatter failures and speed up runs.

Latest finding (most likely root cause)
- Training uses SamplewiseRevNormalization computed from raw batch data.
- Zero-shot extractor was pre-normalizing with stats.yaml, then rollout applied samplewise normalization again.
- This double normalization likely breaks inference.

Latest change made
- Set USE_STATS_NORMALIZATION = False in zero_shot/extractors/shear_flow.py to align with training (samplewise normalization only).
- Rerun started but stopped per request before completion.

Current status
- Field order, padding, BCs, and T_in/T_out match training expectations.
- Large errors persist; double-normalization identified as the main remaining mismatch.
- A rerun with stats normalization disabled was in progress when stopped.

Next recommended steps
- Re-run shear_flow ZSL with USE_STATS_NORMALIZATION=False and check relative MSE.
- If still high, consider using the training pipeline directly (WellDataset/InflatedWellDataset) instead of manual extractor for perfect parity.
