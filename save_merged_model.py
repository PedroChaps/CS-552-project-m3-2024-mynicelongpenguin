from transformers import AutoModelForCausalLM
from peft import PeftModel
import argparse
import torch

smallest_base = "rhysjones/phi-2-orange-v2"
def load_model(args):
        #load base model
        model = AutoModelForCausalLM.from_pretrained(
            smallest_base,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map="auto",
        )
        
        #Load DPO adapter
        model = PeftModel.from_pretrained(model, args.base_model_name)

        #Merge DPO adapter with model
        model = model.merge_and_unload(safe_merge=True, progressbar=True)

        #Load SFT adapter
        model.load_adapter(args.adapter_model_name)

        #Merge SFT adapter with model
        model = model.merge_and_unload(safe_merge=True, progressbar=True)

        return model



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--adapter_model_name", type=str, default=None, help="Base model")
    parser.add_argument("--base_model_name", type=str, default="pleaky2410/TeachersHateChatbots", help="Base model")
    parser.add_argument("--output_dir", type=str, default="None", help="If None, saves on args.adapter_model_name_merged")
    args = parser.parse_args()

    model = load_model(args)

    model.save_pretrained(args.output_dir if args.output_dir != "None" else args.adapter_model_name + "_merged")


if __name__ == "__main__":
    main()