from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch


# TO RUN THIS PROGRAM, ON A CLUSTER CONSOLE, RUN:
# Sinteract -a cs-552 -q cs-552 -g gpu:1 -m 80G -7 00:25:00
# cd /scratch/izar/<username>/project-m3-2024-mynicelongpenguin/
# module load gcc python cuda
# source ~/venvs/sft_mcq/bin/activate
# python save_quantized_model.py

base_model_name = "pleaky2410/TeachersHateChatbotsMCQ"

#########################################################################################################
quantization_config  = BitsAndBytesConfig(
    #TODO: put here your quantization config
)


#TODO: put here the name of the model
outpur_dir = "./model/models/train_sft/outputs/**PUT HERE THE NAME OF THE MODEL**"
#########################################################################################################

def load_model():

    model = AutoModelForCausalLM.from_pretrained(
       base_model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32, 
        quantization_config=quantization_config,
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
