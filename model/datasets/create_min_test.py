import random
import json
from datasets import load_dataset


dataset = load_dataset("json", data_files="test_dataset.jsonl", split="train")
dataset = dataset.shuffle(seed=42).select(range(5000)).remove_columns("related")


with open("test_dataset_min.jsonl", "w") as f:
    for record in dataset:
        f.write(json.dumps(record) + '\n')

