## Key settings overview

Walrus primarily used the isotropic model axis. Baseline models are full replacements using wrappers 
around the code provided by authors on release.

```yaml
defaults:
  - encoder: spacebag_and_vstride_encoder # Use the vstride encoder with a sub-sampled first encoder layer
  - decoder: vstride_decoder 
  - processor: space_time_split # Standard wrapper around space/time factored blocks. Performs operations sequentially.
  - norm_layer: rmsgroupnorm # Pass all dependencies the rmsgroupnorm for use
_target_: walrus.models.IsotropicModel
hidden_dim: 1408
projection_dim: 48 
intermediate_dim: 352 #
processor_blocks: 40
drop_path: .05
groups: 16
max_d: 3
causal_in_time: False
include_d: [2, 3] # Checkpointing with unused parameters has some annoying features which can make it easier to exclude certain decoders.
override_dimensionality: 3 # Force it to treat all data as 3D
jitter_patches: True # Use patch jittering.
gradient_checkpointing_freq: 0 # if >= 1, use gradient checkpointing every # blocks. Encoder/decoder checkpointed when >1.  
```