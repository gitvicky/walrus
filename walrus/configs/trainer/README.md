## Key settings overview

Trainer controls most training settings as the name implies

```yaml
_target_: walrus.trainer.Trainer # Path to trainer
max_epoch: 200 # How many epochs to use - these are fake epochs if max_samples are set lower than data size and recycle is on
val_frequency: 2 # one-step validation every n epochs
rollout_val_frequency: 5 # rollout validation 
short_validation_length: 20 # Until last val, only use this many batches
max_rollout_steps: 60  # Don't rollout longer than this to save time/memory
num_time_intervals: 5 # Break loss logging into this many time intervals 
enable_amp: False.  # Use automatic mixed precision
loss_fn:  # Training loss
  _target_: the_well.benchmark.metrics.MAE
formatter: # Believe this gets overrided
  _target_: hydra.utils.get_class
  path: walrus.data.well_to_multi_transformer.ChannelsFirstWithTimeFormatter
revin:  # Type of normalization to use
  _target_: walrus.trainer.normalization_strat.SamplewiseRevNormalization
  _partial_: True
prediction_type: delta # Options full, delta. Full predicts full field. Delta predicts the difference from previous step. 
grad_acc_steps: 4 # Gradient accumulation rate. Increases batch size, decreases number of optimization steps.
image_validation: True  # Write out plots to disk
video_validation: True # Write out videos to disk
gradient_log_level: 0 # 0 is none, 1 is full norm only, higher not implemented
clip_gradient: 5.0 # Gradient clipping
log_interval: 10 # How often to log training metrics to screen (causes GPU sync)
loss_multiplier: 100. # Multiply the loss by this factor out of underflow paranoia. 
lr_scheduler_per_step: False  # Adjust LR either per step or per "epoch"
validation_suite: # Frame-wise metrics to compute.
  - _target_: the_well.benchmark.metrics.RMSE
  - _target_: the_well.benchmark.metrics.NRMSE
  - _target_: the_well.benchmark.metrics.LInfinity
  - _target_: the_well.benchmark.metrics.VRMSE
  - _target_: the_well.benchmark.metrics.binned_spectral_mse
  - _target_: the_well.benchmark.metrics.PearsonR
validation_trajectory_metrics: # Metrics computed on full trajectory. Can be memory heavy.
  - _target_: the_well.benchmark.metrics.HistogramW1
  - _target_: the_well.benchmark.metrics.WindowedDTW
batch_aggregation_fns: # Note these are strings pointing to function to be resolved by hydra - used to compute aggregate metrics from sample-wise losses
  - torch.mean
  - torch.median
  - torch.std
```