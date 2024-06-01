#! /bin/bash -l
#SBATCH --chdir /scratch/izar/aloureir/project-m3-2024-mynicelongpenguin/model
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 80G
#SBATCH --time 05:00:00
#SBATCH --gres gpu:1
#SBATCH --account cs-552
#SBATCH --qos cs-552

module load gcc python cuda

echo "Loading virtualenv..."

source ~/venvs/venv_evaluator/bin/activate

echo "Going to run..."

python try_model.py --use_template True \
    --idx_question 1243 \
    --adapter_model_name "./models/train_sft/outputs/sft_all_SciQ"  \
    --base_model_name "pleaky2410/TeachersHateChatbots" \
    --is_base_peft True 

echo "Evaluator run"

deactivate
