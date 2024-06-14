from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch


# TO RUN THIS PROGRAM, ON A CLUSTER CONSOLE, RUN:
# Sinteract -a cs-552 -q cs-552 -g gpu:1 -m 80G -t 00:25:00
# cd /scratch/izar/hogenhau/project-m3-2024-mynicelongpenguin/
# module load gcc python cuda
# source ~/venvs/sft_mcq/bin/activate
# python save_quantized_model.py

base_model_name = "pleaky2410/TeachersHateChatbotsMCQ"

#########################################################################################################
# 4-bit quantization configuration
# quantization_config_4bit = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",  # Choose between 'fp4' or 'nf4'
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

# 8-bit quantization configuration
quantization_config_8bit = BitsAndBytesConfig(
    load_in_8bit=True,
    load_in_4bit=False,
    llm_int8_threshold=6.,  # Set your threshold
)

#TODO: put here the name of the model
outpur_dir = "./model/models/train_sft/outputs/8bit_quant_model"
#########################################################################################################

def load_model():

    model = AutoModelForCausalLM.from_pretrained(
       base_model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32, 
        quantization_config=quantization_config_8bit,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    return model, tokenizer


def main():
    
    model, tokenizer = load_model()


    model.save_pretrained(outpur_dir)
    tokenizer.save_pretrained(outpur_dir)


if __name__ == "__main__":
    main()
