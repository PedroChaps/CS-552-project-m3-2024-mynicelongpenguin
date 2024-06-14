#!/bin/bash -l
#SBATCH --chdir /scratch/izar/hogenhau/project-m3-2024-mynicelongpenguin/evaluation_metrics 
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

virtualenv --system-site-packages ~/venvs/eval_venv

source ~/venvs/eval_venv/bin/activate

# upgrade pip the first time you load the environment
pip install --upgrade pip
pip3 install -r requirements.txt

echo "Install Dependency completed"

echo "Going to run the program..."
    
# python3 metrics_calculation.py  \
#     --dataset_name ../model/datasets/test_dataset_min.jsonl \
#     --batch_size 1 \
#     --base_model_name pleaky2410/TeachersHateChatbotsMCQ \
#     --sample_quantity 200 \
#     --output default_model_eval_metrics.json

# python3 metrics_calculation.py  \
#     --dataset_name ../model/datasets/test_dataset_min.jsonl \
#     --batch_size 1 \
#     --adapter_model_name "../model/models/train_sft/outputs/sft_final_model_3998_base" \
#     --sample_quantity 200 \
#     --output 3998_base_model_eval_metrics.json

# python3 metrics_calculation.py  \
#     --dataset_name ../model/datasets/test_dataset_min.jsonl \
#     --batch_size 1 \
#     --adapter_model_name "../model/models/train_sft/outputs/sft_final_model_7997_base" \
#     --sample_quantity 200 \
#     --output 7997_base_model_eval_metrics.json

# python3 metrics_calculation.py  \
#     --dataset_name ../model/datasets/test_dataset_min.jsonl \
#     --batch_size 1 \
#     --adapter_model_name "../model/models/train_sft/outputs/sft_final_model_7997" \
#     --base_model_name "pleaky2410/TeachersHateChatbots" \
#     --is_base_peft True \
#     --sample_quantity 200 \
#     --output 7997_model_eval_metrics.json

# python3 metrics_calculation.py  \
#     --dataset_name ../model/datasets/test_dataset_min.jsonl \
#     --batch_size 1 \
#     --base_model_name "../model/models/train_sft/outputs/4bit_quant_model" \
#     --sample_quantity 200 \
#     --output 4bit_quant_model_eval_metrics.json \
#     --bnb_config_type "nf4"


# python3 metrics_calculation.py  \
#     --dataset_name ../model/datasets/test_dataset_min.jsonl \
#     --batch_size 1 \
#     --base_model_name "../model/models/train_sft/outputs/4bit_nf4_quant_model" \
#     --sample_quantity 200 \
#     --output 4bit_nf4_quant_model_eval_metrics.json \
#     --bnb_config_type "fp4"

# python3 metrics_calculation.py  \
#     --dataset_name ../model/datasets/test_dataset_min.jsonl \
#     --batch_size 1 \
#     --base_model_name "../model/models/train_sft/outputs/8bit_quant_model" \
#     --sample_quantity 200 \
#     --output 8bit_quant_model_eval_metrics.json \
#     --bnb_config_type "8bit"

 
python3 metrics_calculation.py  \
    --dataset_name ../model/datasets/test_dataset_min.jsonl \
    --batch_size 1 \
    --base_model_name "../model/models/train_sft/outputs/sft_final_model_default_merged" \
    --sample_quantity 200 \
    --output sft_final_model_default_merged.json

python3 metrics_calculation.py \
    --get_average_metrics True \
    --metrics_files "default_model_eval_metrics.json,3998_base_model_eval_metrics.json,7997_base_model_eval_metrics.json,7997_model_eval_metrics.json,4bit_quant_model_eval_metrics.json,4bit_nf4_quant_model_eval_metrics.json,8bit_quant_model_eval_metrics.json,sft_final_model_default_merged.json" \
    --output average_metrics.json

echo "Test complete on $(hostname)"
sleep 2

deactivate