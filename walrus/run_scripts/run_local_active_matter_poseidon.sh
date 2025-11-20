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

python train.py distribution=local model=poseidonb name=debugging_next_run_setupv2 trainer=globalnorm_mean_pos trainer.grad_acc_steps=1 server=rusty optimizer=adam optimizer.lr=1.e-4 logger.wandb_project_name="walrus_Debugging" \
            trainer.enable_amp=False trainer.log_interval=5 trainer.clip_gradient=10 data.module_parameters.batch_size=1 data.module_parameters.n_steps_input=1 data.module_parameters.n_steps_output=1   \
            data.module_parameters.max_samples=200 ++model.image_size=[256,512] ++model.num_channels=4 ++model.num_out_channels=4 ++data.module_parameters.recycle_datasets=True ++data.module_parameters.prefetch_field_names=False \
            trainer.short_validation_length=20 ++model.gradient_checkpointing_freq=1 trainer.max_rollout_steps=60 ++trainer.epsilon=1e-5 \
            lr_scheduler=inv_sqrt_w_sqrt_ramps trainer.val_frequency=1 trainer.rollout_val_frequency=1 data.module_parameters.min_dt_stride=1 data.module_parameters.max_dt_stride=1 \
            trainer.prediction_type="full" data=shear_flow_no_pad trainer.max_epoch=101 data_workers=10 auto_resume=False \
            checkpoint=defaults experiment=finetune_example  ++trainer.skip_spectral_metrics=True \
            finetuning_mods=defaults #++finetuning_mods.ape_shape=[32,32,1]  \
