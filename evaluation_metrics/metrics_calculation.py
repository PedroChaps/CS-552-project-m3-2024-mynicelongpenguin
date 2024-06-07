import tokenize
from datasets import load_dataset, load_from_disk, concatenate_datasets, Dataset
import gpt_wrapper
from gpt_wrapper.chat import Chat
import tqdm
import random
import os
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel
import torch
import sys


from bert_score import BERTScorer

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
        model = AutoModelForCausalLM.from_pretrained(
            args.adapter_model_name if args.adapter_model_name else args.base_model_name,
            # quantization_config=bnb_config,
            trust_remote_code=True,
            # torch_dtype=torch.fp32,
            torch_dtype=torch.float32,
            device_map="auto",
        )

        print(model)
        print(model.num_parameters())


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
        diversity_penalty = 0.5,
        repetition_penalty = 1.2,
        no_ngram_repeat_size=5
    )

    input_ids = tokenizer(prompts, return_tensors="pt", return_attention_mask=False).input_ids.to(model.device)

    
    with torch.no_grad():
        answer = model.generate(inputs=input_ids, generation_config=generation_config, tokenizer=tokenizer)
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
    
    def calculate_BERTScore_for_row(row):
        
        answer1 = row["answer1"]
        answer2 = row["answer2"]
        
        scorer = BERTScorer(model_type="bert-base-uncased")
        P, R, F1 = scorer.score([answer1], [answer2])
        
        row["BERTScore_P"] = P[0]
        row["BERTScore_R"] = R[0]
        row["BERTScore_F1"] = F1[0]
        
        return row
    
    
    dataset = dataset.map(lambda x: calculate_BERTScore_for_row(x), num_proc=32)
    
    # Does statistics on the dataset i.e. prints the average values
    # total_P, total_R, total_F1, total_entries = 0, 0, 0, 0
    # for entry in dataset:
    #     total_P += entry["BERTScore_P"]
    #     total_R += entry["BERTScore_R"]
    #     total_F1 += entry["BERTScore_F1"]
    #     total_entries += 1
    
    # print(f"{color.BOLD}### BERTScore ###{color.END}")
    # print(f"{color.BOLD}Total:{color.END}{color.GREEN} {total_entries} entries {color.END}")
    # print(f"{color.BOLD}Average BERTScore P:{color.END}{color.GREEN} {total_P / total_entries}{color.END}")
    # print(f"{color.BOLD}Average BERTScore R:{color.END}{color.GREEN} {total_R / total_entries}{color.END}")
    # print(f"{color.BOLD}Average BERTScore F1:{color.END}{color.GREEN} {total_F1 / total_entries}{color.END}")
    # print()
    
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
    
    # Does statistics on the dataset i.e. prints the average BLEU value
    # total_BLEU, total_entries = 0, 0
    # for entry in dataset:
    #     total_BLEU += entry["BLEU"]
    #     total_entries += 1
    
    # print(f"{color.BOLD}### BLEU ###{color.END}")
    # print(f"{color.BOLD}Total:{color.END}{color.GREEN} {total_entries} entries {color.END}")
    # print(f"{color.BOLD}Average BLEU:{color.END}{color.GREEN} {total_BLEU / total_entries}{color.END}")
    # print()
    
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
    
    # Does statistics on the dataset i.e. prints the average ROUGE values
    # total_ROUGE_1, total_ROUGE_2, total_ROUGE_L, total_entries = 0, 0, 0, 0
    # for entry in dataset:
    #     total_ROUGE_1 += entry["ROUGE-1"]
    #     total_ROUGE_2 += entry["ROUGE-2"]
    #     total_ROUGE_L += entry["ROUGE-L"]
    #     total_entries += 1
    
    # print(f"{color.BOLD}### ROUGE ###{color.END}")
    # print(f"{color.BOLD}Total:{color.END}{color.GREEN} {total_entries} entries {color.END}")
    # print(f"{color.BOLD}Average ROUGE-1:{color.END}{color.GREEN} {total_ROUGE_1 / total_entries}{color.END}")
    # print(f"{color.BOLD}Average ROUGE-2:{color.END}{color.GREEN} {total_ROUGE_2 / total_entries}{color.END}")
    # print(f"{color.BOLD}Average ROUGE-L:{color.END}{color.GREEN} {total_ROUGE_L / total_entries}{color.END}")
    # print()
    
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

    parser.add_argument("--exp_inferences_path", type=str, required=True, help="The path for the file with the expected inferences")
    parser.add_argument("--output", type=str, required=True, help="The file to save the output to")
    parser.add_argument("--batch_size", type=int, default=8, help="The batch size to do inference and evaluation with")
    parser.add_argument("--adapter_model_name", type=str, default=None, help="Base model")
    parser.add_argument("--base_model_name", type=str, default="rhysjones/phi-2-orange-v2", help="Base model")
    parser.add_argument("--is_base_peft", type=bool, default=False, help="Base model")


    args = parser.parse_args()
    
    exp_inferences_dataset = load_dataset("json", data_files=args.exp_inferences_path, split="train")
    
    # It is assumed to have the "question" and "answer" columns
    assert "prompt" in exp_inferences_dataset[0] and "expected" in exp_inferences_dataset[0]
    
    # TODO: uncomment me
    # model, tokenizer = load_sft_model(args)
    
    # Creates an empty dataset that we will be concatenating stuff with
    final_dataset = Dataset.from_dict({"answer1": [], "answer2": []})
        
    
    for i in range(0, len(exp_inferences_dataset), args.batch_size):
        
        real_batch_size = min(args.batch_size, len(exp_inferences_dataset) - i)
        
        # TODO: uncomment me
        # model_answers, expecteds = make_inference(exp_inferences_dataset, range(i, i + real_batch_size), model, tokenizer)
        model_answers, expecteds = ["I like turtles", "fireflies are cool"], ["I like trains", "Olha uma avioneta"]
        
        # Creates a dataset with "answer1" and "answer2" to fit the expected format of the `calculate_...` functions
        dataset = Dataset.from_dict({"answer1": model_answers, "answer2": expecteds})
        
        # Make calculations for this batch
        dataset = calculate_ROUGE(dataset)
        dataset = calculate_BLEU(dataset)
        # TODO uncomment me
        # dataset = calculate_BERTScore(dataset)
        
        # Merge this batch with the final dataset
        final_dataset = concatenate_datasets([final_dataset, dataset])
        
        # Subtracts 2 because of "answer1" and "answer2"
        # n_metrics = len(final_dataset[0]) - 2
        avg_metrics = {key: sum([entry[key] for entry in final_dataset]) / len(final_dataset) for key in final_dataset[0] if key not in ("answer1", "answer2")}
        
        print(f"{color.BOLD}### AVERAGE METRICS ###{color.END}")
        print(f"Processed entries: {len(final_dataset)}")
        for key in avg_metrics:
            print(f"{color.BOLD}{key}{color.END}: {color.GREEN}{avg_metrics[key]}{color.END}")
        

    
    # saves the dataset to disk
    dataset.to_json(args.output)



if __name__ == "__main__":
    main()