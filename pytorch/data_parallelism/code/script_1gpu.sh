#!/bin/bash

#SBATCH --job-name raysgd_1gpu
#SBATCH -D .
#SBATCH --output ../results/1gpu/%j.out
#SBATCH --error ../results/1gpu/%j.err
#SBATCH --ntasks-per-node=1
#SBATCH -c 40
#SBATCH --nodes=1
#SBATCH --gres='gpu:1'
#SBATCH --time 30:00:00
#SBATCH --exclusive
module purge; module load gcc/8.3.0 cuda/10.2 cudnn/7.6.4 nccl/2.4.8 tensorrt/6.0.1 openmpi/4.0.1 atlas/3.10.3 scalapack/2.0.2 fftw/3.3.8 szip/2.1.1 ffmpeg/4.2.1 opencv/4.1.1 python/3.7.4_ML ray/1.4.1
export PYTHONUNBUFFERED=1
ray start --head --num-cpus=10 --num-gpus=1
python multiexperiment.py -g 1
