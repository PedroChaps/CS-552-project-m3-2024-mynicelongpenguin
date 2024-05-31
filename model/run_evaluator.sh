#! /bin/bash -l
#SBATCH --chdir /scratch/izar/aloureir/project-m3-2024-mynicelongpenguin/model
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 80G
#SBATCH --time 08:00:00
#SBATCH --gres gpu:1
#SBATCH --account cs-552
#SBATCH --qos cs-552

module load gcc python cuda

echo "Loading virtualenv..."

source ~/venvs/venv_evaluator/bin/activate

echo "Going to run..."

python evaluator.py

echo "Evaluator run"

deactivate
