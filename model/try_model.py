import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig
import argparse
from datasets import load_dataset
from peft import PeftModel
import random


ANSWER_TEMPLATE = """### EXPLANATION
<explanation>

### ANSWER
<correct_option>"""

QUESTION_TEMPLATE = """### QUESTION
<question>

###OPTIONS
<options>"""

smallest_base = "rhysjones/phi-2-orange-v2"


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
    if args.quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        bnb_config = None  # No quantization

    if args.is_base_peft and args.adapter_model_name is not None:
        model = AutoModelForCausalLM.from_pretrained(
            smallest_base,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map="auto",
            quantization_config=bnb_config
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
            quantization_config=bnb_config
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
            messages.append(tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True))

        # return {"text": texts}
        return {"messages": messages, "correct": samples["correct_option"]}

    dataset = load_dataset("json", data_files=args.dataset_name, split="train")

    dataset = dataset.shuffle(seed=args.seed)

    dataset = dataset.map(get_mcq_options, batched=True, remove_columns=dataset.column_names)

    dataset = dataset.map(apply_template, batched=True, remove_columns=dataset.column_names)

    if args.random_idx:
        idx = random.randint(0, len(dataset))
    
    else:
        idx = args.idx_question

    entry = dataset[idx]["messages"]
    correct = dataset[idx]["correct"]
    return entry, correct, idx


system_prompt = {"role": "system", "content": "You are a helpful EPFL chatbot."}

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--use_template", type=bool, default=True)
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--idx_question", type=int, default=None)
    parser.add_argument("--adapter_model_name", type=str, default=None, help="Base model")
    parser.add_argument("--base_model_name", type=str, default="rhysjones/phi-2-orange-v2", help="Base model")
    parser.add_argument("--is_base_peft", type=bool, default=False, help="Base model")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--dataset_name", type=str, default="datasets/test_dataset.jsonl")
    parser.add_argument("--random_idx", type=bool, default=False)
    parser.add_argument("--quantize", type=bool, default=False)

    args = parser.parse_args()  


    if args.use_template:
        assert (args.random_idx and not args.idx_question) or (not args.random_idx and args.idx_question), "Must provide either random index or index of question (and not both) if using template"

    if args.question and args.use_template:
        print("Question will not be used because using template")

    model, tokenizer = load_sft_model(args)

    print(f"EOS token: {tokenizer.eos_token}\tID: {tokenizer.eos_token_id}")
    print(f"BOS token: {tokenizer.bos_token}\tID: {tokenizer.bos_token_id}")
    print(f"PAD token: {tokenizer.pad_token}\tID: {tokenizer.pad_token_id}")
    print()

    if args.use_template:
        prompt, correct, idx = load_dataset_loss(args, tokenizer)
    else:
        prompt = tokenizer.apply_chat_template([system_prompt, {"role": "user", "content": args.question}], tokenize=False, add_generation_prompt=True)
        idx = None
    
    
    print("### PROMPT#################")
    print(prompt)
    print("###########################")
    print()

    if idx:
        print("### DATASET INDEX#################")
        print(idx)
        print("###########################")
        print()
        print("### CORRECT OPTION#################")
        print(correct)
        print("###########################")
        print()

    input_ids = tokenizer(prompt, return_tensors="pt", return_attention_mask=False).input_ids.to(model.device)

    print("### INPUT IDS ##################")
    print(input_ids)
    print("################################")
    print()
    print("### DECODED INPUT IDS###########")
    text = tokenizer.batch_decode(input_ids, skip_special_tokens=False)[0]
    print(text)
    print("################################")
    print()

    # generation_config = GenerationConfig(
    #     eos_token_id = tokenizer.eos_token_id,
    #     pad_token_id = tokenizer.pad_token_id,
    #     num_beams = 10,
    #     num_beam_groups = 5,
    #     max_new_tokens = 500,
    #     diversity_penalty = 0.5,
    #     repetition_penalty = 1.2,
    # )
    generation_config = GenerationConfig(
        eos_token_id = tokenizer.eos_token_id,
        pad_token_id = tokenizer.pad_token_id,
        num_beams = 10,
        num_beam_groups = 5,
        max_new_tokens = 1023,
        diversity_penalty = 1.0,
        repetition_penalty = 1.2,
        early_stopping=False,
        no_repeat_ngram_size = 5
    )

    # generation_config = GenerationConfig(
    #     eos_token_id = tokenizer.eos_token_id,
    #     pad_token_id = tokenizer.pad_token_id,
    #     do_sample=True,
    #     max_new_tokens = 500,
    #     temperature = 0.7,
    #     top_p = 0.95,
    #     top_k = 50,
    #     repetition_penalty = 1.2,
    # )

    print("### GENERATION CONFIG ###########")
    print(generation_config)
    print("################################")
    print()

    with torch.no_grad():
        answer = model.generate(inputs=input_ids, generation_config=generation_config, tokenizer=tokenizer)
    text = tokenizer.batch_decode(answer, skip_special_tokens=False)[0]

    print("### ANSWER ####################")
    print(text)


if __name__=="__main__":
    main()


    
