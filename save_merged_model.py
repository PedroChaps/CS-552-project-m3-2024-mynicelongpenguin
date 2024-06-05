from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
import torch

smallest_base = "rhysjones/phi-2-orange-v2"
def load_model(args):
        #load ALREADY MERGED DPO model
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map="auto",
        )
        
        #Load SFT adapter
        model = PeftModel.from_pretrained(model, args.adapter_model_name)

        #Merge SFT adapter with DPO merged model
        model = model.merge_and_unload(safe_merge=True, progressbar=True)

        tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)

        return model, tokenizer



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--adapter_model_name", type=str, default=None, help="Adapter model")
    parser.add_argument("--base_model_name", type=str, default="./model/models/train_sft/outputs/DPOMerged", help="Base model")
    parser.add_argument("--output_dir", type=str, default="None", help="If None, saves on args.adapter_model_name_merged")
    args = parser.parse_args()

    model, tokenizer = load_model(args)

    save_path = args.output_dir if args.output_dir != "None" else args.adapter_model_name + "_merged"


    print("Saving model to ", save_path)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)



if __name__ == "__main__":
    main()