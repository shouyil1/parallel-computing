#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=1:00:00
#SBATCH --partition=gpu 
#SBATCH --output=p1_result.out
#SBATCH --gres=gpu:v100:1

module purge

module load cuda/11.6.2  
module load nvidia-hpc-sdk

./p1