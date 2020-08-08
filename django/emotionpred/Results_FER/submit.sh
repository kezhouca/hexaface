#!/bin/bash
#SBATCH --account=def-bgates
#SBATCH --gres=gpu:2        # request GPU "generic resource"
#SBATCH --cpus-per-task=12   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=10G        # memory per node
#SBATCH --time=00-24:00           # time (DD-HH:MM)
#SBATCH --output=out.txt # output file

module load cuda cudnn 
python ./fer_pred.py
