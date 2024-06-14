from datasets import load_dataset, concatenate_datasets, Dataset
import os
import evaluate
from requests import get
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig
from peft import PeftModel
import torch
import sys
import torch

import json

import argparse

smallest_base = "rhysjones/phi-2-orange-v2"

QUESTION_TEMPLATE = """### QUESTION
<question>

###OPTIONS
<options>"""


ANSWER_TEMPLATE = """### EXPLANATION
<explanation>

### ANSWER
<correct_option>"""


def get_average_metrics(datasets, metrics_files):
    def get_avg_per_dataset(dataset):
        avg_metrics = {key: sum([entry[key] for entry in dataset]) / len(dataset) for key in dataset.column_names if key not in ("model_answer", "expected")}
        return avg_metrics
    average_metrics = [get_avg_per_dataset(dataset) for dataset in datasets]
    average_metrics = {metrics_files[i]: average_metrics[i] for i in range(len(metrics_files))}
    return average_metrics



def get_mcq_options(samples):
    """
    Returns a dataset in the format:
    {
        "question": [str],
        "options": [list[str]],
        "correct_option": [str]
        "explanation": [str]
    }
    """
    questions_and_options = samples["question"]
    remove_prefix = len("Question: ")
    remove_suffix = len("\n\nAnswer")

    questions_and_options = [question_and_options[remove_prefix:-remove_suffix] for question_and_options in questions_and_options]

    questions_and_options = [question_and_options.split("\n\nOptions:\n") for question_and_options in questions_and_options]

    questions, options = zip(*questions_and_options)
    questions = list(questions)

    options = list(options)
    options = [opts.split("\n") for opts in options]
    options = [[opt for opt in opts if opt != "" and opt != " "] for opts in options]

    to_remove = ["A)", "B)", "C)", "D)", "E)", "F)", "A. ", "B. ", "C. ", "D. ", "E. ", "F. "]

    for remove in to_remove:
        options = [[opt.replace(remove, "") if opt.startswith(remove) else opt for opt in opts] for opts in options]

    return {"question": questions, "options": options, "explanation": samples["explanation"], "correct_option": samples["answer"]}
    

def create_datasets(args, tokenizer):
    def apply_template(samples):
        system = {"role": "system", "content": "You are a helpful EPFL chatbot."}
        prompts = []
        expected = []

        for i, question in enumerate(samples["question"]):
            prompt = QUESTION_TEMPLATE.replace("<question>", question)
            options=""
            for j, opt in enumerate(samples["options"][i]):
                options += f"{chr(ord('A') + j)}: {opt}\n"

            prompt = prompt.replace("<options>", options)

            conversation = [system]
            conversation.append({"role": "user", "content": prompt})

            prompts.append(tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True))
            
            answer =  ANSWER_TEMPLATE.replace("<explanation>", samples["explanation"][i]).replace("<correct_option>", samples["correct_option"][i])

            conversation.append({"role": "assistant", "content": answer})

            expected.append(tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False))

        # return {"text": texts}
        return {"prompt": prompts, "expected": expected}

    dataset = load_dataset("json", data_files=args.dataset_name, split="train")

    dataset = dataset.shuffle(seed=args.seed)
    if args.sample_quantity:
        dataset = dataset.select(range(args.sample_quantity))


    dataset = dataset.map(get_mcq_options, batched=True, remove_columns=dataset.column_names)
    dataset = dataset.map(apply_template, batched=True, remove_columns=dataset.column_names)

    return dataset

def load_sft_model(args):
    if args.is_base_peft and args.adapter_model_name is not None:
        model = AutoModelForCausalLM.from_pretrained(
            smallest_base,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map="auto",
        )
        
        model = PeftModel.from_pretrained(model, args.base_model_name)

        model = model.merge_and_unload(safe_merge=True, progressbar=True)

        model.load_adapter(args.adapter_model_name)

    else:
        if args.bnb_config_type == 'nf4':
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        elif args.bnb_config_type == 'fp4':
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="fp4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        elif args.bnb_config_type == '8bit':
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.,
            )

        model = AutoModelForCausalLM.from_pretrained(
            args.adapter_model_name if args.adapter_model_name else args.base_model_name,
            quantization_config=bnb_config,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map="auto",
        )

        print(model)

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_model_name if args.adapter_model_name else args.base_model_name)
    
    print(f"EOS token: {tokenizer.eos_token}\tID: {tokenizer.eos_token_id}")
    print(f"BOS token: {tokenizer.bos_token}\tID: {tokenizer.bos_token_id}")
    print(f"PAD token: {tokenizer.pad_token}\tID: {tokenizer.pad_token_id}")
    print(f"CHAT TEMPLATE: {tokenizer.chat_template}")
    print()

    return model, tokenizer


def make_inference(dataset, dataset_idxs, model, tokenizer):
    """
    Generates model answers for a given dataset using the provided model and tokenizer.

    Args:
        dataset: The dataset containing prompts and expected answers.
        dataset_idxs (list): The indices of the dataset to use for inference.
        model: The model used for generating answers.
        tokenizer: The tokenizer used for tokenizing the prompts.

    Returns:
        tuple: A tuple containing the generated model answers and the expected answers.

    """
    prompts = [dataset[dataset_idx]["prompt"] for dataset_idx in dataset_idxs]
    expecteds = [dataset[dataset_idx]["expected"] for dataset_idx in dataset_idxs]

    generation_config = GenerationConfig(
            eos_token_id = tokenizer.eos_token_id,
            pad_token_id = tokenizer.pad_token_id,
            num_beams = 10,
            num_beam_groups = 5,
            max_new_tokens = 500,
            diversity_penalty = 1.0,
            repetition_penalty = 1.2,
            early_stopping=False,
            no_repeat_ngram_size = 5
    )

    with torch.no_grad():
        tokenizer.padding_side = "left"
        input_ids = tokenizer(prompts, return_tensors="pt", return_attention_mask=False, truncation=True, padding=True, max_length=1024).input_ids.to(model.device)
        answer = model.generate(inputs=input_ids, generation_config=generation_config)
        model_answers = tokenizer.batch_decode(answer, skip_special_tokens=True)

    return model_answers, expecteds


CURSOR_UP_ONE = '\x1b[1A' 
ERASE_LINE = '\x1b[2K' 

def delete_last_lines(n=1): 
    for _ in range(n): 
        sys.stdout.write(CURSOR_UP_ONE) 
        sys.stdout.write(ERASE_LINE)     

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


def calculate_COMMET_metric(chosen_dataset, model_outputs):
    ...

def calculate_BERTScore(dataset):

    
    def calculate_BERTScore_for_rows(rows):
        
        answer1 = rows["answer1"]
        answer2 = rows["answer2"]
        
        scorer = evaluate.load("bertscore")
        results = scorer.compute(predictions=answer1, references=answer2, model_type="bert-base-uncased") 
       
        answer = {"BERTScore_P": results["precision"], "BERTScore_R": results["recall"], "BERTScore_F1": results["f1"]} 
        
        return answer
    

    dataset = dataset.map(lambda x: calculate_BERTScore_for_rows(x), batched=True, batch_size = 8)
    
    return dataset
    

def calculate_BLEU(dataset):

    bleu = evaluate.load('bleu')

    def calculate_BLEU_for_row(row):
        
        answer1 = row["answer1"]
        answer2 = row["answer2"]
        
        bleu_val = bleu.compute(predictions=[answer1], references=[answer2])["bleu"]
        
        row["BLEU"] = bleu_val
    
        return row
    
    
    dataset = dataset.map(lambda x: calculate_BLEU_for_row(x), num_proc=32)
    return dataset



def calculate_ROUGE(dataset):

    rouge = evaluate.load('rouge')

    def calculate_ROUGUE_for_row(row):
        
        answer1 = row["answer1"]
        answer2 = row["answer2"]
        
        rouge_vals = rouge.compute(predictions=[answer1], references=[answer2])

        row["ROUGE-1"] = rouge_vals["rouge1"]
        row["ROUGE-2"] = rouge_vals["rouge2"]
        row["ROUGE-L"] = rouge_vals["rougeL"]
    
        return row
    
    
    dataset = dataset.map(lambda x: calculate_ROUGUE_for_row(x), num_proc=32)
    return dataset




def get_merged_dataset(args):
    
    assert os.path.exists(args.file1) and os.path.exists(args.file2)
    assert args.file1.endswith(".jsonl") and args.file2.endswith(".jsonl")
    
    file1 = load_dataset("json", data_files=args.file1, split="train")
    file2 = load_dataset("json", data_files=args.file2, split="train")

    assert len(file1) == len(file2)

    for i in range(len(file2)):        
        assert "question" in file1[i] and "question" in file2[i]
        assert "answer" in file1[i] and "answer" in file2[i]
        assert file1[i]["question"] == file2[i]["question"]

    # Creates a dataset that is a merge of both so the .map() method can be applied to the dataset
    file1 = file1.map(lambda x: {"answer1": x["answer"]}).remove_columns("answer")
    file2 = file2.map(lambda x: {"answer2": x["answer"]}).remove_columns(["answer", "question"])
    
    dataset = concatenate_datasets([file1, file2], axis=1)
    return dataset
    
    

def main():
    
    parser = argparse.ArgumentParser()

    # parser.add_argument("--exp_inferences_path", type=str, required=True, help="The path for the file with the expected inferences")
    parser.add_argument("--output", type=str, required=True, help="The file to save the output to")
    parser.add_argument("--batch_size", type=int, default=8, help="The batch size to do inference and evaluation with")
    parser.add_argument("--adapter_model_name", type=str, default=None, help="Base model")
    parser.add_argument("--base_model_name", type=str, default="rhysjones/phi-2-orange-v2", help="Base model")
    parser.add_argument("--is_base_peft", type=bool, default=False, help="Base model")
    parser.add_argument("--dataset_name", type=str, help="The path for the file with the expected inferences")
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducatibility")
    parser.add_argument("--sample_quantity", type=int, default=None, help="Amount of samples to use for evaluation. If not set, use all")
    parser.add_argument("--get_average_metrics", type=bool, default=False, help="Given metrics file, get average metrics")
    parser.add_argument("--metrics_files", type=str, default=None, help="Comma separated metrics file name")
    parser.add_argument("--bnb_config_type", type=str, choices=['nf4', 'fp4', '8bit'], default='nf4', help="Select the BitsAndBytes configuration type (nf4, fp4, or 8bit)")

    args = parser.parse_args()

    if args.get_average_metrics:
        assert args.metrics_files is not None
        metrics_files = args.metrics_files.split(",")
        datasets = [load_dataset("json", data_files=metrics_file, split="train") for metrics_file in metrics_files]

        avg_metrics = get_average_metrics(datasets, metrics_files) 

        with open(args.output, "w") as f:
            json.dump(avg_metrics, f, indent=4)
        
        return


    model, tokenizer = load_sft_model(args)
    
    # exp_inferences_dataset = load_dataset("json", data_files=args.exp_inferences_path, split="train")
    exp_inferences_dataset = create_datasets(args, tokenizer)
    
    # It is assumed to have the "question" and "answer" columns
    assert "prompt" in exp_inferences_dataset[0] and "expected" in exp_inferences_dataset[0]
    
    model_answers = []
    expecteds = []
    for i in range(0, len(exp_inferences_dataset), args.batch_size):
        real_batch_size = min(args.batch_size, len(exp_inferences_dataset) - i)
        model_answer, expected = make_inference(exp_inferences_dataset, range(i, i + real_batch_size), model, tokenizer)
        model_answers += model_answer
        expecteds += expected
    
    print(f"{len(model_answers) = }")
    print(f"{len(expecteds) = }")

    del model
    del tokenizer
    torch.cuda.empty_cache()

    dataset = Dataset.from_dict({"answer1": model_answers, "answer2": expecteds})

    dataset = calculate_ROUGE(dataset)
    dataset = calculate_BLEU(dataset)
    dataset = calculate_BERTScore(dataset)

    print(f"{len(dataset) = }")
    print(f"{dataset.column_names = }")

    avg_metrics = {key: sum([entry[key] for entry in dataset]) / len(dataset) for key in dataset[0] if key not in ("answer1", "answer2")}


    print(f"{color.BOLD}### AVERAGE METRICS ###{color.END}")
    print(f"Processed entries: {len(dataset)}")
    for key in avg_metrics:
        print(f"{color.BOLD}{key}{color.END}: {color.GREEN}{avg_metrics[key]}{color.END}")


    # saves the dataset to disk
    dataset = dataset.rename_column("answer1", "model_answer")
    dataset = dataset.rename_column("answer2", "expected")
    dataset.to_json(args.output)



if __name__ == "__main__":
    main()