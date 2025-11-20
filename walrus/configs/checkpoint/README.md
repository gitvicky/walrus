## Key settings overview

Examing defaults, we have:

```yaml
_target_: walrus.trainer.checkpoints.CheckPointer # Path to checkpointer object we'll pass these parameters to.
save_dir: checkpoints/${name} # save our checkpoints to this path - this actually gets overridden and is no longer used
load_checkpoint_path: null # <path>/checkpoints/sharded_checkpoint_dir/ - Path if we're loading a sharded checkpoint
coalesced_checkpoint_path: null # <path>/checkpoints/coalesced.pth # Only specify load_checkpoint_path or coalesced_checkpoint_path - coalesced uses a single checkpoint assumed to be on CPU
save_best: True # Whether to track the best checkpoint - also not sure if this flag is actually used
checkpoint_frequency: 20. # How frequently to save long-term checkpoints
align_fields: True # Whether to align the fields in the checkpoint to the model, useful if there are new fields or the data is loaded in a different order
load_chkpt_after_finetuning_expansion: False # Whether to load the coalesced checkpoint after expanding the model for finetuning. If False, the checkpoint is loaded before expansion.
```