#!/bin/bash -l

#SBATCH --nodes=1 # Allocate *at least* 1 node to this job.

#SBATCH --ntasks=1 # Allocate *at most* 1 task for job steps in the job

#SBATCH --cpus-per-task=1 # Each task needs only one CPU

#SBATCH --mem=12G # This particular job won't need much memory
#SBATCH --time=1-00:01:00  # 1 day and 1 minute

#SBATCH --job-name="batch job test"

#SBATCH -p p100_1gpu # You could pick other partitions for other jobs

#SBATCH --wait-all-nodes=1  # Run once all resources are available

#SBATCH --output=output_%j-%N.txt # logging per job and per host in the current directory. Both stdout and stderr are logged.


# sbatch -p p100_4gpu --gres=gpu:2 train_script.sh


# -p argument is for the partition of your choice

# -t argument is for the amount of time you would like to allocate

# --gres=gpu:1 specifies to use 1 GPU for the job. On a multi-GPU node, your job would only be able to see that 1 GPU unless you specify --gpu:# where # is the number of GPUs on the node.

export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPU=1

ARGS="
--output_dir ./flamingo-coco_5
--run_name flamingo-tiny-vitL
--do_train --do_eval
--optim adamw_torch
--learning_rate 0.001 
--warmup_steps 5000
--resume_from_checkpoint /home/yagga004/nlp_final/final_nlp_2/flamingo-coco_5/checkpoint-29572
--lr_scheduler_type constant_with_warmup
--per_device_train_batch_size 16
--per_device_eval_batch_size 64
--gradient_accumulation_steps 1
--evaluation_strategy steps
--eval_steps 1000
--save_strategy epoch
--save_total_limit 2
--log_level info
--num_train_epochs 6
--dataloader_num_workers 0
--dataloader_pin_memory True
--ddp_find_unused_parameters False
"

echo $ARGS

echo "change conda env"
source ~/anaconda3/bin/activate
conda init bash
conda activate conda_flamingo_env_2
cd ~/nlp_final/final_nlp_2

if [ $NUM_GPU == 1 ]; then
    echo "running on a single GPU"
    python ./train.py $ARGS 2>&1 | tee ./output_log_phase_2.txt
else
    echo "running on multiple GPUs"
    torchrun --nproc_per_node $NUM_GPU ~/flamingo_code_test/train_and_eval_combined.py $ARGS
fi
