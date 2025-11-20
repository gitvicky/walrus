#!/bin/bash -l


export OMP_NUM_THREADS=32
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN


# module load python cuda cudnn gcc hdf5
# Activate the virtual environment with all the dependencies
# export MODULEPATH=/mnt/home/gkrawezik/modules/rocky8:$MODULEPATH
# module load cuda/12.4 cudnn/9.1.0.70-cuda12 nccl/2.21.5-1+cuda12.4
source /mnt/home/mmccabe/venvs/mamba_well/bin/activate


# Launch the training script
s
python train.py distribution=local model=isotropic_model name=LocalExample trainer=globalnorm trainer.grad_acc_steps=1 server=rusty optimizer=adam optimizer.lr=1.e-4 logger.wandb_project_name="walrus_Debugging" \
            trainer.enable_amp=False model.gradient_checkpointing_freq=1 trainer.log_interval=5 trainer.clip_gradient=10 data.module_parameters.batch_size=1 data.module_parameters.n_steps_input=6 data.module_parameters.n_steps_output=1   \
            model.projection_dim=48 model.intermediate_dim=352 model.hidden_dim=1408 model.groups=16 model.processor_blocks=40 model.drop_path=0.0 \
            model/processor/space_mixing=full_spatial_attention model.processor.space_mixing.num_heads=16 model.processor.time_mixing.num_heads=16 \
            data.module_parameters.max_samples=2000 ++data.module_parameters.recycle_datasets=False  ++data.module_parameters.prefetch_field_names=True \
            model.causal_in_time=True model.jitter_patches=True  trainer.short_validation_length=20 trainer.max_rollout_steps=60 \
            lr_scheduler=inv_sqrt_w_sqrt_ramps trainer.val_frequency=5 trainer.rollout_val_frequency=5 data.module_parameters.min_dt_stride=1 data.module_parameters.max_dt_stride=1 \
            trainer.prediction_type="delta" data=active_matter trainer.max_epoch=51 data_workers=10 model.override_dimensionality=0 auto_resume=False \
            checkpoint=defaults experiment=defaults ++model.use_periodic_fixed_jitter=True ++model.input_field_drop=0 ++trainer.skip_spectral_metrics=True \
            finetuning_mods=defaults 
