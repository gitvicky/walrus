#!/bin/bash -l
#SBATCH --time=36:00:00
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=250gb
#SBATCH -J EVAL_Walrus_example
#SBATCH --output=eval_walrus_example_-%j.log
#SBATCH -C a100-80gb

export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN


# module load python cuda cudnn gcc hdf5
# Activate the virtual environment with all the dependencies
export MODULEPATH=/mnt/home/gkrawezik/modules/rocky8:$MODULEPATH
module load cuda/12.4 cudnn/9.1.0.70-cuda12 nccl/2.21.5-1+cuda12.4
source /mnt/home/mmccabe/venvs/mamba_well/bin/activate


# Launch the training script
# Folder structures defined by the train script can enter validation just by pointing to a config and weight folder. The rest of the settings are telling the run to validate differently than during training.
srun python train.py --config-path="/path/to/folder/containing/extended_config.yaml/" --config-name="extended_config.yaml" ++distribution.distribution_type=local ++validation_mode=True ++folder_override="/path/to/parent_experiment/folder/with/weights/Walrus_ft_noape_euler_multi_quadrants_openBC_realglobalnorm-euler-delta-Isotr\[Space-Adapt-\]-AdamW-0.0001/finetune/0/" "++trainer.validation_suite=[{_target_:the_well.benchmark.metrics.NRMSE},{_target_:the_well.benchmark.metrics.VRMSE},{_target_:the_well.benchmark.metrics.PearsonR}]"   "++trainer.validation_trajectory_metrics=[]"   "++trainer.batch_aggregation_fns=[torch.mean,torch.median,torch.std]" "++data.module_parameters.max_rollout_steps=200"  "++data.module_parameters.start_rollout_valid_output_at_t=17" "++trainer.max_rollout_steps=200" "++data.well_base_path=/mnt/home/polymathic/ceph/the_well/datasets/" 
