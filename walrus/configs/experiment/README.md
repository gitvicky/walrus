## Key settings overview

Mostly self-explanatory:

```yaml
# @package _global_  States to add these to the global namespace
auto_resume: True # Automatically resume if default experiment_folder and checkpoint exist
finetune: True # Treat existing path as pretrained model to finetune in new path
folder_override: "" # Override the folder name for the experiment - considered "resume" if folder has chpts
checkpoint_override: "" # Override search path for a resume checkpoint - considered "resume" if populated
config_override: /mnt/home/polymathic/ceph/MPPX_logging/platinum_checkpoints/final_base_model/extended_config.yaml # Override the search path for a config file
validation_mode: False # Run full training.
frozen_components: # If resuming, these will be automatically imported from the checkpoint config. "all" means use exact settings.
  - model 

```