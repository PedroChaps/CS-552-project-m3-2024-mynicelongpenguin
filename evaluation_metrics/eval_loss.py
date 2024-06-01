import torch
from torch.utils.data import DataLoader
import argparse
from datasets import load_dataset
from transformers import (
    set_seed
)
from tqdm import tqdm

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

##################################################################
# DEBUGGING
def dprint(*args, **kwargs):
    if True:
        print("[DEBUG eval_loss.py]", *args, **kwargs)
##################################################################

seed = 42

ANSWER_TEMPLATE = """### EXPLANATION
<explanation>

### ANSWER
<correct_option>"""

QUESTION_TEMPLATE = """### QUESTION
<question>

###OPTIONS
<options>"""


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

def load_sft_model(args):
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
    
    dprint(f"EOS token: {tokenizer.eos_token}\tID: {tokenizer.eos_token_id}")
    dprint(f"BOS token: {tokenizer.bos_token}\tID: {tokenizer.bos_token_id}")
    dprint(f"PAD token: {tokenizer.pad_token}\tID: {tokenizer.pad_token_id}")
    dprint(f"CHAT TEMPLATE: {tokenizer.chat_template}")
    dprint()

    return model, tokenizer

def load_dataset_loss(args, tokenizer):
    def apply_template(samples):
        system = {"role": "system", "content": "You are a helpful EPFL chatbot."}
        messages = []

        for i, question in enumerate(samples["question"]):
            prompt = QUESTION_TEMPLATE.replace("<question>", question)
            options=""
            for j, opt in enumerate(samples["options"][i]):
                options += f"{chr(ord('A') + j)}: {opt}\n"

            prompt = prompt.replace("<options>", options)

            conversation = [system]
            conversation.append({"role": "user", "content": prompt})
            
            answer =  ANSWER_TEMPLATE.replace("<explanation>", samples["explanation"][i]).replace("<correct_option>", samples["correct_option"][i])

            conversation.append({"role": "assistant", "content": answer})

            messages.append(tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False))

        # return {"text": texts}
        return {"messages": messages}

    dataset = load_dataset("json", data_files=args.dataset_name, split="train")

    dataset = dataset.shuffle(seed=args.seed)
    if args.sample_quantity:
        dataset = dataset.select(range(args.sample_quantity))


    dataset = dataset.map(get_mcq_options, batched=True, remove_columns=dataset.column_names)

    dataset = dataset.map(apply_template, batched=True, remove_columns=dataset.column_names)

    return dataset


def calculate_losses(model, tokenizer, dataset, args):
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    total_loss = 0
    num_batches = 0

    progress_bar = tqdm(total=len(dataloader), desc="Eval")
    for i, batch in enumerate(dataloader):
        tokenized_batch = tokenizer(batch['messages'], padding=True, truncation=True, return_tensors='pt', max_length=2048)

        inputs = tokenized_batch.input_ids.to(model.device)
        attention_mask = tokenized_batch.attention_mask.to(model.device)

        #set pad tokens in labels to -100 so they are masked and do not count towards loss
        labels = torch.where(inputs == tokenizer.pad_token_id, -100, inputs)

        with torch.no_grad():
            outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
        
        loss = outputs.loss
        total_loss += loss.item()
        num_batches += 1

        if i % 10 == 0:
            progress_bar.update(10)
            print(f"Batch {i} loss: {loss.item()}")
        
    
    return total_loss / num_batches

    

def main():
    set_seed(seed)
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=4, help='Batch size to be used for inference')
    parser.add_argument("--dataset_name", type=str, help="Dataset for training")
    parser.add_argument("--sample_quantity", type=int, default=None, help="Number of entries to sample")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--adapter_model_name", type=str, default=None, help="Base model")
    parser.add_argument("--base_model_name", type=str, default="rhysjones/phi-2-orange-v2", help="Base model")
    parser.add_argument("--output_file", type=str, default="output.txt", help="Output file")

    args = parser.parse_args()

    print("ARGS")
    print(args)
    print()
    print()

    model, tokenizer = load_sft_model(args)

    dataset = load_dataset_loss(args, tokenizer)


    avg_loss = calculate_losses(model, tokenizer, dataset, args)

    print("Average Loss:", avg_loss)

    with open(args.output_file, "w") as f:
        f.write(str(avg_loss))


if __name__ == "__main__":
    main()