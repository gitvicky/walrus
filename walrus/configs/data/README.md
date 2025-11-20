## Key settings overview

Examing defaults, we have:

```yaml
defaults:
  - field_index_map_override: bc_only_override # Before populating field_to_index map from data, start with these fixed mappings
  - transform: random_rotation90_train # Apply rotation augmentation
well_base_path: /mnt/gpuxl/polymathic/the_well/datasets/ # Usually overrided by server
wandb_data_name: "well_allmain_only" # Name associated with data in wandb logs and experiment folder
module_parameters: # Parameters used to instantiate data module. 
  _target_: walrus.data.MixedWellDataModule. # Path to data module
  batch_size: 8. # Default batch size
  n_steps_input: 10 # Default context length
  n_steps_output: 1 # Default number of predictions at a time
  min_dt_stride: 1 # Random stride minimum
  max_dt_stride: 1 # Random stride maximum
  max_samples: 2000 # Time in fake epochs (aka validation intervals since datasets can be too large for true epochs to be practice)
  well_dataset_info:  # Describes what data to include and where to find it. No path means this is assuming Well-structure
    active_matter: # Well structured data with known name - dataset will look for it relative to well_base_path.
      include_filters: [] # Only include files with one of these strings in name
      exclude_filters: [] # Exclude files with these strings in name
    acoustic_scattering_maze:
      include_filters: []
      exclude_filters: []
      field_transforms:
        density: torch.zeros_like # Apply this function to field with name density in this dataset
    supernova_explosion_128:
      include_filters: []
      exclude_filters: []
      step_downsample_factor: .5 # Sample half the context length
      batch_downsample_factor: .5 # Sample half the batch size
      field_transforms:
        density: torch.log10 
        temperature: torch.log10
    flowbench_FPO_NS_2D_512x128_harmonics: # Non-well data. operates based on explicit path. 
      include_filters: []
      exclude_filters: []
      path: /mnt/gpuxl/polymathic/WellFormattedExternalData/flowbench/flowbench_FPO_NS_2D_512x128_harmonics # Look here for this data transformed into Well format
```