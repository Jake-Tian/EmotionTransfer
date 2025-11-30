#!/bin/bash
#SBATCH --job-name=sft
#SBATCH --mail-user=yztian25@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=/research/d7/gds/yztian25/EmotionTransfer/output.txt
#SBATCH --gres=gpu:2
#SBATCH --reser=jcheng_gpu_301
#SBATCH --qos=gpu
#SBATCH --account=gpu
#SBATCH -c 12
#SBATCH -p gpu_24h

bash qwen_train/inference/scripts/run_inference_multiple_samples.sh