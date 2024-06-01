#!/bin/bash -l
#SBATCH --chdir /scratch/izar/chaparro/project-m3-2024-mynicelongpenguin/evaluation_metrics 
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 80G
#SBATCH --time 20:00:00
#SBATCH --gres gpu:1
#SBATCH --account cs-552
#SBATCH --qos cs-552
#SBATCH --reservation cs-552


module load gcc python
module load cuda


# You only need to create this virtualenv once
# Feel free to replace the name "course_py-3.10" with your own environemnt name
# virtualenv --system-site-packages ~/venvs/mnlp2

# Activate the virtualenv everytime before you run the job
# Make sure the name matches the one you created
source ~/venvs/sft_mcq/bin/activate


# upgrade pip the first time you load the environment
# pip install --upgrade pip


# Only when you need to update any packages you already installed
# pip3 install torch==2.1.1 --index-url https://download.pytorch.org/whl/cu118
# pip3 install -r ./train_dpo/requirements.txt
# pip3 install -U git+https://github.com/huggingface/trl


echo "Install Dependency completed"

echo "Going to run the program..."

python3 eval_loss.py \
    --adapter_model_name "/scratch/izar/chaparro/project-m3-2024-mynicelongpenguin/model/models/train_sft/outputs/sft_model_1999" \
    --dataset_name "/scratch/izar/chaparro/project-m3-2024-mynicelongpenguin/model/datasets/test_dataset.jsonl" \
    --batch_size 8 \
    --output_file "sft_model_1999_test_loss.jsonl" \
    --sample_quantity 7000

echo "Going to run the program..."

python3 eval_loss.py \
    --adapter_model_name "/scratch/izar/chaparro/project-m3-2024-mynicelongpenguin/model/models/train_sft/outputs/sft_model_1999_base" \
    --dataset_name "/scratch/izar/chaparro/project-m3-2024-mynicelongpenguin/model/datasets/test_dataset.jsonl" \
    --batch_size 8 \
    --output_file "sft_model_1999_base_test_loss.jsonl" \
    --sample_quantity 7000

echo "Going to run the program..."

python3 eval_loss.py \
    --adapter_model_name "/scratch/izar/chaparro/project-m3-2024-mynicelongpenguin/model/models/train_sft/outputs/sft_model_3998" \
    --dataset_name "/scratch/izar/chaparro/project-m3-2024-mynicelongpenguin/model/datasets/test_dataset.jsonl" \
    --batch_size 8 \
    --output_file "sft_model_3998_test_loss.jsonl" \
    --sample_quantity 7000

echo "Going to run the program..."

python3 eval_loss.py \
    --adapter_model_name "/scratch/izar/chaparro/project-m3-2024-mynicelongpenguin/model/models/train_sft/outputs/sft_model_3998_base" \
    --dataset_name "/scratch/izar/chaparro/project-m3-2024-mynicelongpenguin/model/datasets/test_dataset.jsonl" \
    --batch_size 8 \
    --output_file "sft_model_3998_base_test_loss.jsonl" \
    --sample_quantity 7000
    

echo "Test complete on $(hostname)"
sleep 2

deactivate