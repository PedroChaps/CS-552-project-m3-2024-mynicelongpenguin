#!/bin/bash -l
#SBATCH --chdir /scratch/izar/aloureir/project-m3-2024-mynicelongpenguin/model/models
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 80G
#SBATCH --time 20:00:00
#SBATCH --gres gpu:1
#SBATCH --account cs-552
#SBATCH --qos cs-552


module load gcc python
module load cuda


SAMPLE=3000


# You only need to create this virtualenv once
# Feel free to replace the name "course_py-3.10" with your own environemnt name
# virtualenv --system-site-packages ~/venvs/sft_mcq

# Activate the virtualenv everytime before you run the job
# Make sure the name matches the one you created
source ~/venvs/sft_mcq/bin/activate

# huggingface-cli login --token $HF_TOKEN --add-to-git-credential
# huggingface-cli login --token $HF_TOKEN 

# upgrade pip the first time you load the environment
# pip install --upgrade pip


# Only when you need to update any packages you already installed
# pip3 install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118
# pip3 install -r ./train_sft_mcq/requirements.txt
# pip3 install -U git+https://github.com/huggingface/trl


echo "Install Dependency completed"

echo "Going to run the program..."


    # --dataset_name "/scratch/izar/aloureir/project-m3-2024-mynicelongpenguin/model/datasets/Transformed_DeepMind_Algebra_QA_with_explanation.jsonl" \
    # --sample_quantity 10000 \
python3 train_sft/sft/train_sft.py --lr 1e-5 \
    --dataset_name "/scratch/izar/aloureir/project-m3-2024-mynicelongpenguin/model/datasets/Transformed_SciQ_Dataset_with_explanation.jsonl" \
    --output_dir "/scratch/izar/aloureir/project-m3-2024-mynicelongpenguin/model/models/train_sft/outputs/sft_model_try" \
    --effective_batch_size 32 \
    --adapter_model_name "pleaky2410/TeachersHateChatbots" \


echo "Test complete on $(hostname)"
sleep 2

deactivate
