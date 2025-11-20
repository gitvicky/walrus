## Configuration settings

This repository uses [Hydra](https://hydra.cc) for orchestration. Hydra uses a hierarchy of yaml files and command line overrides
and composes them through OmegaConf. This takes a bit of getting used to, so we have a quick overview of the important settings. For 
example, here is our baseline file with some comments. 

```yaml
defaults:
  - _self_
  - trainer: defaults # use "defaults" in trainer subdirectory
  - optimizer: adam # use "adam" in trainer subdirectory
  - lr_scheduler: inv_sqrt_w_linear_ramps # Same pattern
  - model: isotropic_model
  - data: all_2d
  - experiment: defaults
  - server: ??? # Force user to specify server
  - distribution: local
  - logger: wandb
  - checkpoint: defaults
  - finetuning_mods: defaults
data_workers: 14 # Data worker is set here because this was set before we had gotten used to hydra
name: default_name # Project name, used for 
finetune: False # Treat this as finetuning
automatic_setup: True # Use automatic pathway for logging, constructing artifacts, folders and saving outputs
```

