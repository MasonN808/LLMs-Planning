#!/bin/bash

# SLURM settings for the job submission
#SBATCH --job-name=mamba-0           # Name of the job
#SBATCH --gres=gpu:1             # Request one GPU
#SBATCH --mem=8G                # Memory allocated
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks=1                # Number of tasks
#SBATCH --time=1-00:00:00           # Maximum run time of the job (set to 3 days)
#SBATCH --qos=scavenger           # Quality of Service of the job

# Activate python environment, if you use one (e.g., conda or virtualenv)
source .venv/bin/activate

export OPENAI_API_KEY='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'

export FAST_DOWNWARD='/nas/ucb/mason/LLMs-Planning/planner_tools/downward-main'

export PR2='/nas/ucb/mason/LLMs-Planning/planner_tools/PR2'

export VAL='/nas/ucb/mason/LLMs-Planning/planner_tools/VAL'

BASE_SCRIPT="plan-bench/llm_plan_pipeline.py"

srun -N1 -n1 python3 $BASE_SCRIPT --task t1 --config blocksworld_3 --engine mamba

wait


