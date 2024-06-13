from datasets import load_dataset, load_from_disk, concatenate_datasets
import gpt_wrapper
from gpt_wrapper.chat import Chat
import tqdm
import random
import os

import argparse


gpt_wrapper.api_base = "http://mnlp-backend-938795011.eu-central-1.elb.amazonaws.com"
gpt_wrapper.api_key = "175accd9-4b3d-42a7-998e-85600fa3d1f4"

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


def answer_questions(args):
    
    # ⚠️ It assumes both files have the same number of inferences and are aligned
    # ⚠️ It assumes the data has the columns "question" and "answer"
    
    # assert os.path.exists(args.file1) and os.path.exists(args.file2)
    # assert args.file1.endswith(".jsonl") and args.file2.endswith(".jsonl")
    
    dataset = load_dataset("json", data_files=args.file, split="train")
    # file2 = load_dataset("json", data_files=args.file2, split="train")

    # assert len(file1) == len(file2)

    # for i in range(len(file2)):        
    #     assert "question" in file1[i] and "question" in file2[i]
    #     assert "answer" in file1[i] and "answer" in file2[i]
    #     assert file1[i]["question"] == file2[i]["question"]

    # Creates a dataset that is a merge of both so the .map() method can be applied to the dataset
    # file1 = file1.map(lambda x: {"answer1": x["answer"]}).remove_columns("answer")
    # file2 = file2.map(lambda x: {"answer2": x["answer"]}).remove_columns(["answer", "question"])
    
    # dataset = concatenate_datasets([file1, file2], axis=1)
    
    new_column = [None] * len(dataset)
    dataset = dataset.add_column("chatgpt_answer", new_column)

    # print(f"{dataset[0] = }")
    # print(f"{dataset = }")
    
    template = '''
    Given the following question, output the correct choice and only the correct choice.

    Question: {question}

    Correct Choice: 
    '''

    # The function to be passed to map()
    def answer_question(entry):
        
        chat = Chat.create(f"ask_ChatGPT_which_is_better_{random.randint(0, 999999999999)}")
        prompt = template.format(question=entry["question"])
        message = str(chat.ask(content=prompt))
        
        gpt_answer = "A"
        for i in range(len(message)-1, 0, -1):
            if message[i] in ["A", "B", "C", "D", "E", "F", "G"]:
                gpt_answer = message[i]
                break
            
        entry["chatgpt_answer"] = gpt_answer
        
        return entry
    
    dataset_rated = dataset.map(lambda x: answer_question(x), num_proc=320)
    dataset_rated.to_json(args.output)
    
    return dataset_rated
    

def print_statistics(args, dataset_rated):
    
    num_correct = 0
    
    for entry in dataset_rated:
        
        if entry["chatgpt_answer"] == entry["answer"]:
            num_correct += 1
            
    accuracy = num_correct / len(dataset_rated)
    
    print(f"{color.BOLD}Accuracy: {accuracy:.2f}{color.END}")
    print(f"{color.BOLD}Number of correct answers: {num_correct}{color.END}")
    print(f"{color.BOLD}Total number of questions: {len(dataset_rated)}{color.END}")


def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--file", type=str, required=True, help="The file with the test data")
    parser.add_argument("--output", type=str, required=True, help="The file to save the output to")

    args = parser.parse_args()
    
    # print(args)
    # print(args.file1)
    
    dataset_rated = answer_questions(args)
    print_statistics(args, dataset_rated)
    
    
    
if __name__ == "__main__":
    main()